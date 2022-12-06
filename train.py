import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from torch.cuda.amp import GradScaler, autocast
from transformers import (
    AutoTokenizer, 
    AutoModelForQuestionAnswering,
    default_data_collator,
    get_linear_schedule_with_warmup
)
from torch.utils.tensorboard import SummaryWriter
from datasets import load_dataset
import evaluate
from tqdm import tqdm

from arg import get_train_args
from process import SquadPreprocess, SquadPostprocess
import util


def train(args):
    # setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    name = args.pretrain_model.replace('/', '-')
    save_dir = util.get_save_dir('save', name, training=True)
    log = util.get_logger(save_dir, name)
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))
    tb = SummaryWriter(save_dir)
    util.set_random_seed(args.seed)

    # get raw dataset
    train_data_path = f'data/{args.train_split}-v2.0.hf.json'
    dev_data_path = f'data/{args.dev_split}-v2.0.hf.json'
    log.info(f'Loading dataset from {train_data_path} and {dev_data_path}')
    raw_dataset = load_dataset('json', data_files={args.train_split: train_data_path, args.dev_split: dev_data_path})

    # pre-process the dataset
    log.info('Pre-processing dataset')
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model)
    preprocessor = SquadPreprocess(tokenizer)
    train_dataset = preprocessor.process_dataset(raw_dataset[args.train_split], is_train=True)
    dev_dataset, offset_mappings, example_ids = preprocessor.process_dataset(raw_dataset[args.dev_split], is_train=False)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=default_data_collator
    )
    dev_dataloader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=default_data_collator
    )

    # get model
    log.info(f'Loading model from {args.pretrain_model}')
    model = AutoModelForQuestionAnswering.from_pretrained(args.pretrain_model)
    if args.trained_weight_path is not None:
        log.info(f'Loading trained weights from {args.trained_weight_path}')
        model.load_state_dict(torch.load(args.trained_weight_path))
    model = nn.DataParallel(model, gpu_ids)
    model = model.to(device)
    model.train()
    ema = util.EMA(model, args.ema_decay)

    # get checkpoint saver
    saver = util.CheckpointSaver(save_dir,
                                 max_checkpoints=args.max_checkpoints,
                                 metric_name=args.metric_name,
                                 maximize_metric=args.maximize_metric,
                                 log=log)

    # get optimizer and scheduler
    params = list(model.named_parameters())
    no_decay_layers = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optim_grouped_params = [
        {'params': [p for n, p in params if not any(nd in n for nd in no_decay_layers)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in params if any(nd in n for nd in no_decay_layers)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optim_grouped_params, lr=args.lr)
    total_steps = args.num_epochs * len(train_dataloader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    # training
    log.info('Training...')
    steps_till_eval = args.eval_steps
    best_metric_val = 0.0 if args.maximize_metric else 1e10
    postprocessor = SquadPostprocess(tokenizer)
    metric = evaluate.load('squad_v2')
    scaler = GradScaler() if args.use_fp16 else None
    epoch = 0
    step = 0

    while epoch < args.num_epochs:
        epoch += 1
        log.info(f'Starting epoch {epoch}...')
        with torch.enable_grad(), tqdm(desc='Train', total=len(train_dataloader)) as pbar:
            for batch in train_dataloader:
                # setup
                batch = {k: v.to(device) for k, v in batch.items()}
                optimizer.zero_grad()

                if not args.use_fp16:
                    # forward
                    loss = model(**batch)['loss'].sum()
                    # backward
                    loss.backward()
                    clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                else:
                    # fp16 counterpart
                    with autocast():
                        loss = model(**batch)['loss'].sum()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()

                scheduler.step()
                ema(model, step)

                # log info
                step += 1
                pbar.update(1)
                loss_val = loss.item()
                pbar.set_postfix(epoch=epoch, NLL=loss_val)
                tb.add_scalar('train/NLL', loss_val, step)
                tb.add_scalar('train/LR', optimizer.param_groups[0]['lr'], step)

                steps_till_eval -= 1
                if steps_till_eval <= 0:
                    steps_till_eval = args.eval_steps

                    # evaluate and save checkpoint
                    log.info(f'Evaluating at step {step}...')
                    ema.assign(model)
                    results = validate(
                        model, dev_dataloader, postprocessor, metric, raw_dataset[args.dev_split],
                        dev_dataset, offset_mappings, example_ids, args.batch_size_postpro, log, device
                    )
                    saver.save(step, model, results[args.metric_name], device)
                    ema.resume(model)

                    # log to console
                    results_str = ', '.join(f'{k}: {v:.2f}' for k, v in results.items())
                    log.info(f'Dev {results_str}')

                    # log to TensorBoard
                    log.info('Visualizing in TensorBoard...')
                    for k, v in results.items():
                        tb.add_scalar(f'dev/{k}', v, step)
                    
                    # early stop check
                    if (args.maximize_metric and results[args.metric_name] > best_metric_val) \
                        or (not args.maximize_metric and results[args.metric_name] < best_metric_val):
                        early_stop_cnt = 0
                        best_metric_val = results[args.metric_name]
                        log.info(f'Best {args.metric_name}: {best_metric_val:.2f}')
                    else:
                        early_stop_cnt += 1
                        log.info(f'Not best, wait for early stop: {early_stop_cnt}/{args.early_stop_patience}')
                    if early_stop_cnt >= args.early_stop_patience:
                        log.info('Early stopped')
                        return
    log.info('Reached maximum epochs, stopped')
    

def validate(
    model, dataloader, postprocessor, metric,
    raw_dataset, pro_dataset, offset_mappings,
    example_ids, batch_size_postpro, log, device
):
    model.eval()

    # get predicted start/end logits
    log.info('Forwarding model to get start/end logits')
    start_logits = []
    end_logits = []
    loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Test', total=len(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            loss += output['loss'].sum().item()
            start_logits.extend(output['start_logits'].tolist())
            end_logits.extend(output['end_logits'].tolist())
    loss /= len(dataloader)

    # post-process logits to get final answers
    log.info('Post-processing logits to get final answers')

    ## (this can be slow, even with multiple CPUs)
    predictions = postprocessor.process_predictions(
        raw_dataset, pro_dataset,
        offset_mappings, example_ids,
        start_logits, end_logits,
        batch_size=batch_size_postpro
    )

    # compute the metrics
    log.info('Computing metrics')
    formatted_predictions = [{'id': k, 'prediction_text': v, 'no_answer_probability': 0.0} for k, v in predictions.items()]
    references = [{'id': ex['id'], 'answers': ex['answers']} for ex in raw_dataset]
    results = metric.compute(predictions=formatted_predictions, references=references)

    model.train()
    return {'NLL': loss, 'EM': results['exact'], 'F1': results['f1']}


if __name__ == '__main__':
    train(get_train_args())

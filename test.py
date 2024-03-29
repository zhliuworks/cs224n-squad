import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator
from datasets import load_dataset
import evaluate
from tqdm import tqdm

from arg import get_test_args
from process import SquadPreprocess, SquadPostprocess
import util


def test(args):
    # setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    name = args.pretrain_model.replace('/', '-')
    save_dir = util.get_save_dir('save', name, training=False)
    log = util.get_logger(save_dir, name)
    device, gpu_ids = util.get_available_devices()
    args.batch_size *= max(1, len(gpu_ids))

    # get raw dataset
    data_path = f'data/{args.split}-v2.0.hf.json'
    log.info(f'Loading dataset from {data_path}')
    raw_dataset = load_dataset('json', data_files={args.split: data_path})

    # if predictions file not provided
    if args.pred_path is None:
        # pre-process the dataset
        log.info('Pre-processing dataset')
        tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model)
        preprocessor = SquadPreprocess(tokenizer)
        pro_dataset, offset_mappings, example_ids = preprocessor.process_dataset(raw_dataset[args.split], is_train=False)
        dataloader = DataLoader(
            pro_dataset,
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
        model.eval()

        # get predicted start/end logits
        log.info('Forwarding model to get start/end logits')
        start_logits = []
        end_logits = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc='Forward', total=len(dataloader)):
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(**batch)
                start_logits.extend(output['start_logits'].tolist())
                end_logits.extend(output['end_logits'].tolist())

        # post-process logits to get final answers
        log.info('Post-processing logits to get final answers')
        postprocessor = SquadPostprocess(tokenizer)

        ## (this can be slow, even with multiple CPUs)
        predictions = postprocessor.process_predictions(
            raw_dataset[args.split], pro_dataset,
            offset_mappings, example_ids,
            start_logits, end_logits,
            batch_size=args.batch_size_postpro
        )

        # save the predictions
        pred_save_path = os.path.join(save_dir, f'preds_{args.split}.json')
        log.info(f'Predictions saved to {pred_save_path}')
        with open(pred_save_path, 'w') as fw:
            json.dump(predictions, fw)
    
    # just load predictions file
    else:
        log.info(f'Predictions loaded from {args.pred_path}')
        with open(args.pred_path, 'r') as fr:
            predictions = json.load(fr)

    # compute the metrics
    log.info('Computing metrics')
    metric = evaluate.load('squad_v2')
    formatted_predictions = [{'id': k, 'prediction_text': v, 'no_answer_probability': 0.0} for k, v in predictions.items()]
    references = [{'id': ex['id'], 'answers': ex['answers']} for ex in raw_dataset[args.split]]
    results = metric.compute(predictions=formatted_predictions, references=references)
    log.info('F1: {:.2f}'.format(results['f1']))
    log.info('EM: {:.2f}'.format(results['exact']))


if __name__ == '__main__':
    test(get_test_args())

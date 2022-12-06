import argparse


def get_common_args(desc):
    parser = argparse.ArgumentParser(desc)
    add = parser.add_argument
    add('--pretrain_model', type=str, required=True, help='pretrained huggingface model released on hub, e.g., albert-base-v2')
    add('--trained_weight_path', type=str, default=None, help='trained model weights to load')
    add('--batch_size', type=int, default=16, help='batch size per GPU for data loading')
    add('--gpu_ids', type=str, default='0', help='cuda visible devices')
    add('--num_workers', type=int, default=32, help='number of subprocesses for data loading')
    add('--batch_size_postpro', type=int, default=1024, help='batch size for parallel post-processing')
    return parser
    

def get_train_args():
    parser = get_common_args('Train a model on SQuAD 2.0')
    add = parser.add_argument
    add('--train_split', type=str, default='train', choices=['train', 'dev', 'test'])
    add('--dev_split', type=str, default='dev', choices=['train', 'dev', 'test'])
    add('--num_epochs', type=int, default=10)
    add('--lr', type=float, default=1e-4)
    add('--weight_decay', type=float, default=1e-4)
    add('--warmup_ratio', type=float, default=0.05, help='ratio of learning rate warmup steps')
    add('--max_grad_norm', type=float, default=5.0, help='maximum gradient norm for gradient clipping')
    add('--ema_decay', type=float, default=0.999, help='decay rate for exponential moving average of parameters')
    add('--metric_name', type=str, default='F1', choices=['NLL', 'EM', 'F1'], help='name of dev metric to determine best checkpoint')
    add('--max_checkpoints', type=int, default=3, help='maximum number of checkpoints to keep on disk')
    add('--eval_steps', type=int, default=2000, help='number of steps between successive evaluations')
    add('--early_stop_patience', type=int, default=5, help='number of evaluations to stop training if the metric is not better')
    add('--use_fp16', action='store_true', help='use automatic mixed precision to accelerate training')
    add('--seed', type=int, default=0)
    args = parser.parse_args()

    if args.metric_name == 'NLL':
        # Best checkpoint is the one that minimizes negative log-likelihood
        args.maximize_metric = False
    elif args.metric_name in ('EM', 'F1'):
        # Best checkpoint is the one that maximizes EM or F1
        args.maximize_metric = True
    else:
        raise ValueError(f'Unrecognized metric name: "{args.metric_name}"')

    return args


def get_test_args():
    parser = get_common_args('Test a trained model on SQuAD 2.0')
    add = parser.add_argument
    add('--split', type=str, default='dev', choices=['train', 'dev', 'test'])
    add('--pred_path', type=str, default=None, help='load predictions to just compute the metrics')
    args = parser.parse_args()
    return args

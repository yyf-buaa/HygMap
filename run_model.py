import argparse

from libcity.pipeline import run_model
from libcity.utils import add_other_args


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='STRL', help='the name of task')
    parser.add_argument('--model', type=str, default='HOME', help='the name of model')
    parser.add_argument('--dataset', type=str, default='cd', help='the name of dataset')
    parser.add_argument('--config_file', type=str, default=None, help='the file name of config file')
    parser.add_argument('--exp_id', type=str, default=None, help='id of experiment')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--output_dim', type=int, default=128, help='output_dim')
    add_other_args(parser, 'hgi')
    args = parser.parse_args()
    dict_args = vars(args)
    other_args = {key: val for key, val in dict_args.items() if key not in [
        'task', 'model', 'dataset', 'config_file', 'saved_model', 'train'] and
        val is not None}
    run_model(task=args.task, model_name=args.model, dataset_name=args.dataset,
              config_file=args.config_file, saved_model=args.saved_model,
              train=args.train, other_args=other_args)

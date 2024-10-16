"""
store the arguments can be modified by the user
"""
import argparse

hgi_arguments = {
    "gpu": {
        "type": "bool",
        "default": None,
        "help": "whether use gpu"
    },
    "gpu_id": {
        "type": "int",
        "default": None,
        "help": "the gpu id to use"
    },
    "train_rate": {
        "type": "float",
        "default": None,
        "help": "the train set rate"
    },
    "eval_rate": {
        "type": "float",
        "default": None,
        "help": "the validation set rate"
    },
    "batch_size": {
        "type": "int",
        "default": None,
        "help": "the batch size"
    },
    "learning_rate": {
        "type": "float",
        "default": None,
        "help": "learning rate"
    },
    "max_epoch": {
        "type": "int",
        "default": None,
        "help": "the maximum epoch"
    },
    "train": {
        "type": "bool",
        "default": True,
        "help": "whether re-train model if the model is trained before"
    },
    "saved_model": {
        "type": "bool",
        "default": True,
        "help": "whether save the trained model"
    },
    "dataset_class": {
        "type": "str",
        "default": None,
        "help": "the dataset class name"
    },
    "executor": {
        "type": "str",
        "default": None,
        "help": "the executor class name"
    },
    "evaluator": {
        "type": "str",
        "default": None,
        "help": "the evaluator class name"
    },
    "learner": {
        "type": "str",
        "default": None,
        "help": "type of optimizer"
    },
    "grad_accmu_steps": {
        "type": "int",
        "default": None,
        "help": "learning rate"
    },
    "lr_decay": {
        "type": "bool",
        "default": None,
        "help": "whether to use le_scheduler"
    },
    "lr_scheduler": {
        "type": "str",
        "default": None,
        "help": "type of lr_scheduler"
    },
    "lr_eta_min": {
        "type": "float",
        "default": None,
        "help": "min learning rate"
    },
    "lr_warmup_epoch": {
        "type": "int",
        "default": None,
        "help": "warm-up epochs"
    },
    "lr_warmup_init": {
        "type": "float",
        "default": None,
        "help": "initial lr for warm-up"
    },
    "t_in_epochs": {
        "type": "bool",
        "default": None,
        "help": "whether update lr epoch by epoch(True) / batch by batch(False)"
    },
    "clip_grad_norm": {
        "type": "bool",
        "default": None,
        "help": "Whether to use gradient cropping"
    },
    "max_grad_norm": {
        "type": "float",
        "default": None,
        "help": "Maximum gradient"
    },
    "use_early_stop": {
        "type": "bool",
        "default": None,
        "help": "Whether to use early-stop"
    },
    "patience": {
        "type": "int",
        "default": None,
        "help": "early-stop epochs"
    },
    "log_every": {
        "type": "int",
        "default": None,
        "help": "Frequency of logging epoch by epoch"
    },
    "log_batch": {
        "type": "int",
        "default": None,
        "help": "Frequency of logging batch by batch"
    },
    "load_best_epoch": {
        "type": "bool",
        "default": None,
        "help": "Whether to load best model for test"
    },
    "road_region_ratio": {
        "type": "float",
        "default": None,
        "help": "road_region_ratio"
    },
    "region_city_ratio": {
        "type": "float",
        "default": None,
        "help": "region_city_ratio"
    },
    "road_cl_ratio": {
        "type": "float",
        "default": None,
        "help": "road_cl_ratio"
    },
    "region_cl_ratio": {
        "type": "float",
        "default": None,
        "help": "region_cl_ratio"
    },
    "tau": {
        "type": "float",
        "default": None,
        "help": "tau"
    },
    "attn_heads": {
        "type": "int",
        "default": None,
        "help": "attn_heads"
    },
    "dropout": {
        "type": "float",
        "default": None,
        "help": "dropout"
    },
    "emb_dim": {
        "type": "int",
        "default": None,
        "help": "emb_dim"
    },
    "num_layers": {
        "type": "int",
        "default": None,
        "help": "num_layers"
    },
    "edge_agu": {
        "type": "bool",
        "default": None,
        "help": "edge_agu"
    },
    "fea_agu": {
        "type": "bool",
        "default": None,
        "help": "fea_agu"
    },
    "add_road2region": {
        "type": "bool",
        "default": None,
        "help": "add_road2region"
    },
    "add_emb_layer": {
        "type": "bool",
        "default": None,
        "help": "add_emb_layer"
    },
    "intra_road2region": {
        "type": "bool",
        "default": None,
        "help": "intra_road2region"
    },
}


def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool value expected.')


def str2float(s):
    if isinstance(s, float):
        return s
    try:
        x = float(s)
    except ValueError:
        raise argparse.ArgumentTypeError('float value expected.')
    return x


def add_other_args(parser, args_type):
    data = hgi_arguments
    for arg in data:
        if data[arg]['type'] == 'int':
            parser.add_argument('--{}'.format(arg), type=int,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'bool':
            parser.add_argument('--{}'.format(arg), type=str2bool,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'float':
            parser.add_argument('--{}'.format(arg), type=str2float,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'str':
            parser.add_argument('--{}'.format(arg), type=str,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'list of int':
            parser.add_argument('--{}'.format(arg), nargs='+', type=int,
                                default=data[arg]['default'], help=data[arg]['help'])
        elif data[arg]['type'] == 'list of str':
            parser.add_argument('--{}'.format(arg), nargs='+', type=str,
                                default=data[arg]['default'], help=data[arg]['help'])

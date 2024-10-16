import datetime
import logging
import math
import os
import sys
import argparse
import random

import networkx as nx
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

def set_random_seed(seed):
    """
    重置随机数种子

    Args:
        seed(int): 种子数
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def str2bool(s):
    if isinstance(s, bool):
        return s
    if s.lower() in ('yes', 'true'):
        return True
    elif s.lower() in ('no', 'false'):
        return False
    else:
        raise argparse.ArgumentTypeError('bool value expected.')


def batch_ts_to_start_time(ts_array):
    for i in range(len(ts_array)):
        dt = datetime.datetime.utcfromtimestamp(ts_array[i])
        ts_array[i] = dt.hour
    return ts_array

def load_graph(rel_path):
    rel = pd.read_csv(rel_path)
    G = nx.DiGraph()
    for i, row in tqdm(rel.iterrows(), total=rel.shape[0]):
        prev_id = row.origin_id
        curr_id = row.destination_id
        G.add_edge(prev_id, curr_id)
    return G


def concat_csv_path(base_path, file_name: str, added_str=''):
    if file_name[-4:] == '.csv':
        file_name = file_name[:-4]
    return os.path.join(base_path, file_name + added_str + ".csv")


def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def set_random_seed(seed):
    """
    重置随机数种子

    Args:
        seed(int): 种子数
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def get_local_time():
    """
    获取时间

    Return:
        datetime: 时间
    """
    cur = datetime.datetime.now()
    cur = cur.strftime('%b-%d-%Y_%H-%M-%S')
    return cur


def get_logger(exp_id, model, task, dataset, log_dir, name=None):
    """
    获取Logger对象

    Returns:
        Logger: logger
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    log_filename = '{}-{}-{}-{}-{}.log'.format(exp_id, model, task, dataset, get_local_time())
    logfilepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)

    log_level = 'INFO'

    if log_level.lower() == 'info':
        level = logging.INFO
    elif log_level.lower() == 'debug':
        level = logging.DEBUG
    elif log_level.lower() == 'error':
        level = logging.ERROR
    elif log_level.lower() == 'warning':
        level = logging.WARNING
    elif log_level.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfilepath)
    file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('Log directory: %s', log_dir)
    return logger


def top_k(loc_pred, loc_true, topk):
    """
    count the hit numbers of loc_true in topK of loc_pred, used to calculate Precision, Recall and F1-score,
    calculate the reciprocal rank, used to calcualte MRR,
    calculate the sum of DCG@K of the batch, used to calculate NDCG

    Args:
        loc_pred: (batch_size * output_dim)
        loc_true: (batch_size * 1)
        topk:

    Returns:
        tuple: tuple contains:
            hit (int): the hit numbers \n
            rank (float): the sum of the reciprocal rank of input batch \n
            dcg (float): dcg
    """
    assert topk > 0, "top-k ACC评估方法：k值应不小于1"
    loc_pred = torch.FloatTensor(loc_pred)  # (batch_size * output_dim)
    val, index = torch.topk(loc_pred, topk, 1)  # dim=1上的前k大的值以及下标
    index = index.numpy()  # (batch_size * topk)  也就是预测的最高概率的topk个类别
    hit = 0
    rank = 0.0
    dcg = 0.0
    for i, p in enumerate(index):  # i->batch, p->(topk,)
        target = loc_true[i]  # 第i个数据的真实类别
        if target in p:
            hit += 1  # 命中一次
            rank_list = list(p)
            rank_index = rank_list.index(target)  # 真值在预测值中排的顺序
            # rank_index is start from 0, so need plus 1
            rank += 1.0 / (rank_index + 1)
            dcg += 1.0 / np.log2(rank_index + 2)
    return hit, rank, dcg


class Scheduler:
    """ Parameter Scheduler Base Class
    A scheduler base class that can be used to schedule any optimizer parameter groups.

    Unlike the builtin PyTorch schedulers, this is intended to be consistently called
    * At the END of each epoch, before incrementing the epoch count, to calculate next epoch's value
    * At the END of each optimizer update, after incrementing the update count, to calculate next update's value

    The schedulers built on this should try to remain as stateless as possible (for simplicity).

    This family of schedulers is attempting to avoid the confusion of the meaning of 'last_epoch'
    and -1 values for special behaviour. All epoch and update counts must be tracked in the training
    code and explicitly passed in to the schedulers on the corresponding step or step_update call.

    Based on ideas from:
     * https://github.com/pytorch/fairseq/tree/master/fairseq/optim/lr_scheduler
     * https://github.com/allenai/allennlp/tree/master/allennlp/training/learning_rate_schedulers
    """

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 param_group_field: str,
                 noise_range_t=None,
                 noise_type='normal',
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=None,
                 initialize: bool = True):
        self.optimizer = optimizer
        self.param_group_field = param_group_field
        self._initial_param_group_field = f"initial_{param_group_field}"
        if initialize:
            for i, group in enumerate(self.optimizer.param_groups):
                if param_group_field not in group:
                    raise KeyError(f"{param_group_field} missing from param_groups[{i}]")
                group.setdefault(self._initial_param_group_field, group[param_group_field])
        else:
            for i, group in enumerate(self.optimizer.param_groups):
                if self._initial_param_group_field not in group:
                    raise KeyError(f"{self._initial_param_group_field} missing from param_groups[{i}]")
        self.base_values = [group[self._initial_param_group_field] for group in self.optimizer.param_groups]
        self.metric = None  # any point to having this for all?
        self.noise_range_t = noise_range_t
        self.noise_pct = noise_pct
        self.noise_type = noise_type
        self.noise_std = noise_std
        self.noise_seed = noise_seed if noise_seed is not None else 42
        self.update_groups(self.base_values)

    def state_dict(self):
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}

    def load_state_dict(self, state_dict):
        self.__dict__.update(state_dict)

    def get_epoch_values(self, epoch):
        return None

    def get_update_values(self, num_updates):
        return None

    def step(self, epoch, metric=None):
        self.metric = metric
        values = self.get_epoch_values(epoch)
        if values is not None:
            values = self._add_noise(values, epoch)
            self.update_groups(values)

    def step_update(self, num_updates, metric=None):
        self.metric = metric
        values = self.get_update_values(num_updates)
        if values is not None:
            values = self._add_noise(values, num_updates)
            self.update_groups(values)

    def update_groups(self, values):
        if not isinstance(values, (list, tuple)):
            values = [values] * len(self.optimizer.param_groups)
        for param_group, value in zip(self.optimizer.param_groups, values):
            param_group[self.param_group_field] = value

    def _add_noise(self, lrs, t):
        if self.noise_range_t is not None:
            if isinstance(self.noise_range_t, (list, tuple)):
                apply_noise = self.noise_range_t[0] <= t < self.noise_range_t[1]
            else:
                apply_noise = t >= self.noise_range_t
            if apply_noise:
                g = torch.Generator()
                g.manual_seed(self.noise_seed + t)
                if self.noise_type == 'normal':
                    while True:
                        # resample if noise out of percent limit, brute force but shouldn't spin much
                        noise = torch.randn(1, generator=g).item()
                        if abs(noise) < self.noise_pct:
                            break
                else:
                    noise = 2 * (torch.rand(1, generator=g).item() - 0.5) * self.noise_pct
                lrs = [v + v * noise for v in lrs]
        return lrs


class CosineLRScheduler(Scheduler):

    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_initial: int,
                 t_mul: float = 1.,
                 lr_min: float = 0.,
                 decay_rate: float = 1.,
                 warmup_t=0,
                 warmup_lr_init=0,
                 warmup_prefix=False,
                 cycle_limit=0,
                 t_in_epochs=True,
                 noise_range_t=None,
                 noise_pct=0.67,
                 noise_std=1.0,
                 noise_seed=42,
                 initialize=True) -> None:
        super().__init__(
            optimizer, param_group_field="lr",
            noise_range_t=noise_range_t, noise_pct=noise_pct, noise_std=noise_std, noise_seed=noise_seed,
            initialize=initialize)

        assert t_initial > 0
        assert lr_min >= 0
        self._logger = logging.getLogger()
        if t_initial == 1 and t_mul == 1 and decay_rate == 1:
            self._logger.warning("Cosine annealing scheduler will have no effect on the learning "
                                 "rate since t_initial = t_mul = eta_mul = 1.")
        self.t_initial = t_initial
        self.t_mul = t_mul
        self.lr_min = lr_min
        self.decay_rate = decay_rate
        self.cycle_limit = cycle_limit
        self.warmup_t = warmup_t
        self.warmup_lr_init = warmup_lr_init
        self.warmup_prefix = warmup_prefix
        self.t_in_epochs = t_in_epochs
        if self.warmup_t:
            self.warmup_steps = [(v - warmup_lr_init) / self.warmup_t for v in self.base_values]
            super().update_groups(self.warmup_lr_init)
        else:
            self.warmup_steps = [1 for _ in self.base_values]

    def _get_lr(self, t):
        if t < self.warmup_t:
            lrs = [self.warmup_lr_init + t * s for s in self.warmup_steps]
        else:
            if self.warmup_prefix:
                t = t - self.warmup_t

            if self.t_mul != 1:
                i = math.floor(math.log(1 - t / self.t_initial * (1 - self.t_mul), self.t_mul))
                t_i = self.t_mul ** i * self.t_initial
                t_curr = t - (1 - self.t_mul ** i) / (1 - self.t_mul) * self.t_initial
            else:
                i = t // self.t_initial
                t_i = self.t_initial
                t_curr = t - (self.t_initial * i)

            gamma = self.decay_rate ** i
            lr_min = self.lr_min * gamma
            lr_max_values = [v * gamma for v in self.base_values]

            if self.cycle_limit == 0 or (self.cycle_limit > 0 and i < self.cycle_limit):
                lrs = [
                    lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * t_curr / t_i)) for lr_max in
                    lr_max_values
                ]
            else:
                lrs = [self.lr_min for _ in self.base_values]

        return lrs

    def get_epoch_values(self, epoch: int):
        if self.t_in_epochs:
            return self._get_lr(epoch)
        else:
            return None

    def get_update_values(self, num_updates: int):
        if not self.t_in_epochs:
            return self._get_lr(num_updates)
        else:
            return None

    def get_cycle_length(self, cycles=0):
        if not cycles:
            cycles = self.cycle_limit
        cycles = max(1, cycles)
        if self.t_mul == 1.0:
            return self.t_initial * cycles
        else:
            return int(math.floor(-self.t_initial * (self.t_mul ** cycles - 1) / (1 - self.t_mul)))


bert_arguments = {
    "gpu": {
        "type": "bool",
        "default": True,
        "help": "whether use gpu"
    },
    "gpu_id": {
        "type": "int",
        "default": None,
        "help": "the gpu id to use"
    },
    "seed": {
        "type": "int",
        "default": None,
        "help": "random seed"
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
    "data_time": {
        "type": "str",
        "default": None,
        "help": "data time"
    },
    "roadmap_path": {
        "type": "str",
        "default": None,
        "help": "roadmap path"
    },
    "vocab_path": {
        "type": "str",
        "default": None,
        "help": "built vocab model path with bert-vocab"
    },
    "min_freq": {
        "type": "int",
        "default": None,
        "help": "Minimum frequency of occurrence of road segments"
    },
    "merge": {
        "type": "bool",
        "default": None,
        "help": "Whether to merge 3 dataset to get vocab"
    },
    "d_model": {
        "type": "int",
        "default": None,
        "help": "hidden size of transformer model"
    },
    "mlp_ratio": {
        "type": "int",
        "default": None,
        "help": "The ratio of FNN layer dimension to d_model"
    },
    "lape_dim": {
        "type": "int",
        "default": None,
        "help": "laplacian vector hidden size of transformer model"
    },
    "random_flip": {
        "type": "bool",
        "default": None,
        "help": "whether random flip laplacian vector when train"
    },
    "n_layers": {
        "type": "int",
        "default": None,
        "help": "number of layers"
    },
    "attn_heads": {
        "type": "int",
        "default": None,
        "help": "number of attention heads"
    },
    "seq_len": {
        "type": "int",
        "default": None,
        "help": "maximum sequence len"
    },
    "future_mask": {
        "type": "bool",
        "default": None,
        "help": "Whether to mask the future timestep, True is single-direction attention, False for double-direction"
    },
    "pretrain_road_emb": {
        "type": "str",
        "default": None,
        "help": "Path of pretrained road emb vector"
    },
    "dropout": {
        "type": "float",
        "default": None,
        "help": " The dropout probability for all fully connected layers in the embeddings, encoder, and pooler."
    },
    "attn_drop": {
        "type": "float",
        "default": None,
        "help": "The dropout ratio for the attention probabilities."
    },
    "drop_path": {
        "type": "float",
        "default": None,
        "help": "dropout of encoder block"
    },
    "masking_ratio": {
        "type": "float",
        "default": None,
        "help": "mask ratio of input"
    },
    "masking_mode": {
        "type": "str",
        "default": None,
        "help": "mask all dim together or mask dim separate"
    },
    "distribution": {
        "type": "str",
        "default": None,
        "help": "random mask or makov mask i.e. independent mask or continuous mask"
    },
    "avg_mask_len": {
        "type": "int",
        "default": None,
        "help": "average mask length for makov mask i.e. continuous mask"
    },
    "type_ln": {
        "type": "str",
        "default": None,
        "help": "pre-norm or post-norm"
    },
    "test_every": {
        "type": "int",
        "default": None,
        "help": "Frequency of testing on the test set"
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
    "l2_reg": {
        "type": "bool",
        "default": None,
        "help": "Whether to use L2 regularization"
    },
    "initial_ckpt": {
        "type": "str",
        "default": None,
        "help": "Path of the model parameters to be loaded"
    },
    "unload_param": {
        "type": "list of str",
        "default": None,
        "help": "unloaded pretrain parameters"
    },
    "add_cls": {
        "type": "bool",
        "default": None,
        "help": "Whether add CLS in BERT"
    },
    "pooling": {
        "type": "str",
        "default": None,
        "help": "Trajectory embedding pooling method"
    },
    "pretrain_path": {
        "type": "str",
        "default": None,
        "help": "Path of pretrained model"
    },
    "freeze": {
        "type": "bool",
        "default": None,
        "help": "Whether to freeze the pretrained BERT"
    },
    "topk": {
        "type": "list of int",
        "default": None,
        "help": "top-k value for classification evaluator"
    },
    "n_views": {
        "type": "int",
        "default": None,
        "help": "number of views for contrastive learning"
    },
    "similarity": {
        "type": "str",
        "default": None,
        "help": "similarity of different representations for contrastive learning"
    },
    "temperature": {
        "type": "float",
        "default": None,
        "help": "temperature of nt-xent loss for contrastive learning"
    },
    "predict_t": {
        "type": "bool",
        "default": None,
        "help": "whether predict time when pre-train"
    },
    "contra_ratio": {
        "type": "float",
        "default": None,
        "help": "contrastive loss ratio"
    },
    "contra_loss_type": {
        "type": "str",
        "default": None,
        "help": "contrastive loss type, i.e. simclr, simsce, consert"
    },
    "mtm_ratio": {
        "type": "float",
        "default": None,
        "help": "mtm(predict time task) loss ratio"
    },
    "mlm_ratio": {
        "type": "float",
        "default": None,
        "help": "mlm(predict location task) loss ratio"
    },
    "cutoff_row_rate": {
        "type": "float",
        "default": None,
        "help": "cutoff_row_rate for data argument"
    },
    "cutoff_column_rate": {
        "type": "float",
        "default": None,
        "help": "cutoff_column_rate for data argument"
    },
    "cutoff_random_rate": {
        "type": "float",
        "default": None,
        "help": "cutoff_random_rate for data argument"
    },
    "sample_rate": {
        "type": "float",
        "default": None,
        "help": "sample_rate for data argument"
    },
    "data_argument1": {
        "type": "list of str",
        "default": None,
        "help": "data argument methods for view1"
    },
    "data_argument2": {
        "type": "list of str",
        "default": None,
        "help": "data argument methods for view2"
    },
    "classify_label": {
        "type": "str",
        "default": None,
        "help": "classify label for downstream task, vflag, usrid"
    },
    "use_pack": {
        "type": "bool",
        "default": None,
        "help": "whether use pack method in base rnn model"
    },
    "cluster_kinds": {
        "type": "int",
        "default": None,
        "help": "cluster kinds"
    },
    "add_time_in_day": {
        "type": "bool",
        "default": None,
        "help": "whether use time_in_day emb"
    },
    "add_day_in_week": {
        "type": "bool",
        "default": None,
        "help": "whether use day_in_week emb"
    },
    "add_pe": {
        "type": "bool",
        "default": None,
        "help": "whether use position emb"
    },
    "add_usr": {
        "type": "bool",
        "default": None,
        "help": "whether use usr emb"
    },
    "add_lap": {
        "type": "bool",
        "default": None,
        "help": "whether use laplacian emb"
    },
    "add_degree": {
        "type": "bool",
        "default": None,
        "help": "whether use graph degree emb"
    },
    "use_gcn_emb": {
        "type": "bool",
        "default": None,
        "help": "whether use gcn to deal the road feature"
    },
    "use_gcn_and_pretrain": {
        "type": "bool",
        "default": None,
        "help": "whether use gcn to deal the road feature and load pretrained road emb vector"
    },
    "gcn_channel": {
        "type": "list of int",
        "default": None,
        "help": "dim of gcn layers"
    },
    "max_diffusion_step": {
        "type": "int",
        "default": None,
        "help": "K-order Chebyshev polynomial approximation"
    },
    "filter_type": {
        "type": "str",
        "default": None,
        "help": "Type of Laplace matrix in graph convolution, range in (laplacian, random_walk, dual_random_walk)"
    },
    "roadnetwork": {
        "type": "str",
        "default": None,
        "help": "road network dataset"
    },
    "geo_file": {
        "type": "str",
        "default": None,
        "help": "road network dataset"
    },
    "rel_file": {
        "type": "str",
        "default": None,
        "help": "road network dataset"
    },
    "bidir_adj_mx": {
        "type": "bool",
        "default": None,
        "help": "whether use bi-dir adj_mx forced"
    },
    "cluster_data_path": {
        "type": "str",
        "default": None,
        "help": "test data name for cluster"
    },
    "query_data_path": {
        "type": "str",
        "default": None,
        "help": "test data name for similarity-search"
    },
    "detour_data_path": {
        "type": "str",
        "default": None,
        "help": "test detour data name for similarity-search"
    },
    "origin_big_data_path": {
        "type": "str",
        "default": None,
        "help": "test database name of for similarity-search"
    },
    "sim_select_num": {
        "type": "int",
        "default": None,
        "help": "num of trajectories in similarity-search task"
    },
    "baseline": {
        "type": "bool",
        "default": None,
        "help": "whether use bert-baseline model"
    },
    "finetune_baseline": {
        "type": "bool",
        "default": None,
        "help": "whether finetune bert-baseline model"
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


def add_other_args(parser):
    data = bert_arguments
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



import os
import json
import torch
import random

from libcity.config import ConfigParser
from libcity.data import get_dataset
from libcity.utils import get_executor, get_model, get_logger, ensure_dir, set_random_seed


def run_model(task=None, model_name=None, dataset_name=None, config_file=None,
              saved_model=True, train=True, other_args=None):
    """
    Args:
        task(str): task name
        model_name(str): model name
        dataset_name(str): dataset name
        config_file(str): config filename used to modify the pipeline's
            settings. the config file should be json.
        saved_model(bool): whether to save the model
        train(bool): whether to train the model
        other_args(dict): the rest parameter args, which will be pass to the Config
    """
    # load config
    config = ConfigParser(task, model_name, dataset_name, config_file, saved_model, train, other_args)
    exp_id = config.get('exp_id', None)
    if exp_id is None:
        # Make a new experiment ID
        exp_id = int(random.SystemRandom().random() * 1000000)
        config['exp_id'] = exp_id
    # logger
    logger = get_logger(config)
    logger.info('Begin pretrain-pipeline, task={}, model_name={}, dataset_name={}, exp_id={}'.
                format(str(task), str(model_name), str(dataset_name), str(exp_id)))
    logger.info(config.config)
    # seed
    seed = config.get('seed', 0)
    set_random_seed(seed)
    # 加载数据集
    dataset = get_dataset(config)
    # 转换数据，并划分数据集
    train_data, _, _ = dataset.get_data()
    data_feature = dataset.get_data_feature()
    # 加载执行器
    model_cache_file = config.get('model_cache_file', './libcity/cache/{}/model_cache/{}_{}_{}.pt'.format(
        exp_id, exp_id, model_name, dataset_name))
    model = get_model(config, data_feature)
    executor = get_executor(config, model, data_feature)

    # 训练和评估
    if train:
        executor.train(train_data, _, _)
        if saved_model:
            executor.save_model(model_cache_file)
        executor.evaluate(train_data)
    else:
        load_epoch = config.get('load_epoch', None)
        if load_epoch is not None:
            executor.load_model_with_epoch(config['load_epoch'])
        # executor.train(train_data, _, _)
        # if saved_model:
        #     executor.save_model(model_cache_file)
        executor.evaluate(train_data)



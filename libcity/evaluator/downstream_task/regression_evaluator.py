import datetime
import json
import logging
import os
import pickle

import pandas as pd
import torch
from matplotlib import pyplot as plt

from libcity.evaluator.abstract_evaluator import AbstractEvaluator
from libcity.evaluator.downstream_task import loss
from libcity.utils.downstream_utils import batch_ts_to_start_time, ensure_dir


class RegressionEvaluator(AbstractEvaluator):

    def __init__(self, args):
        self.config = vars(args)
        self.metrics = self.config.get('metrics', ["MAE", "RMSE", "MAPE", "R2", "EVAR"])  # 评估指标, 是一个 list
        self.allowed_metrics = ['MAE', 'MSE', 'RMSE', 'MAPE', 'masked_MAE',
                                'masked_MSE', 'masked_RMSE', 'masked_MAPE', 'R2', 'EVAR']
        self.save_modes = self.config.get('save_modes', ['csv', 'json'])
        self.result = {}  # 每一种指标的结果
        self.intermediate_result = {}  # 每一种指标每一个batch的结果
        self._check_config()
        self._logger = logging.getLogger()

    def _check_config(self):
        if not isinstance(self.metrics, list):
            raise TypeError('Evaluator type is not list')
        for metric in self.metrics:
            if metric not in self.allowed_metrics:
                raise ValueError('the metric {} is not allowed in RegressionEvaluator'.format(str(metric)))

    def collect(self, batch):
        """
        收集一 batch 的评估输入

        Args:
            batch(dict): 输入数据，字典类型，包含两个Key:(y_true, y_pred):
                batch['y_true']: (batch_size, 1)
                batch['y_pred']: (batch_size, 1)
        """
        if not isinstance(batch, dict):
            raise TypeError('evaluator.collect input is not a dict of user')
        y_true = batch['y_true'].cpu()  # tensor
        y_pred = batch['y_pred'].cpu()  # tensor
        hop = batch['hop'].cpu()
        length = batch['length'].cpu()
        tlist = batch['tlist'][:, 0].cpu()
        start_hour = batch_ts_to_start_time(tlist)
        if y_true.shape != y_pred.shape:
            raise ValueError("batch['y_true'].shape is not equal to batch['y_pred'].shape")

        for metric in self.metrics + ['true', 'pred', 'length', 'hop', 'start_hour']:
            if metric not in self.intermediate_result:
                self.intermediate_result[metric] = []

        self.intermediate_result['true'].append(y_true)
        self.intermediate_result['pred'].append(y_pred)
        self.intermediate_result['length'].append(length)
        self.intermediate_result['hop'].append(hop)
        self.intermediate_result['start_hour'].append(start_hour)

    def calc_metric(self, y_pred, y_true, metric):
        if metric == 'masked_MAE':
            return loss.masked_mae_torch(y_pred, y_true, 0).item()
        elif metric == 'masked_MSE':
            return loss.masked_mse_torch(y_pred, y_true, 0).item()
        elif metric == 'masked_RMSE':
            return loss.masked_rmse_torch(y_pred, y_true, 0).item()
        elif metric == 'masked_MAPE':
            return loss.masked_mape_torch(y_pred, y_true, 0).item()
        elif metric == 'MAE':
            return loss.masked_mae_torch(y_pred, y_true).item()
        elif metric == 'MSE':
            return loss.masked_mse_torch(y_pred, y_true).item()
        elif metric == 'RMSE':
            return loss.masked_rmse_torch(y_pred, y_true).item()
        elif metric == 'MAPE':
            return loss.masked_mape_torch(y_pred, y_true).item()
        elif metric == 'R2':
            return loss.r2_score_torch(y_pred, y_true).item()
        elif metric == 'EVAR':
            return loss.explained_variance_score_torch(y_pred, y_true).item()

    def plot_result(self, y_true, y_pred, length, hop, start_hour, save_path):
        # res_data = pd.DataFrame({'true': y_true, 'pred': y_pred, 'length': length, 'hop': hop, 'start_hour': start_hour})
        res_data = pd.DataFrame(
            {'true': y_true.squeeze(), 'pred': y_pred.squeeze(), 'length': length.squeeze(), 'hop': hop.squeeze(),
             'start_hour': start_hour.squeeze()})
        hop_group = dict(list(res_data.groupby(by='hop')))
        hop_res = {}
        for h, v in hop_group.items():
            curr_hop = h
            curr_pred = v['pred'].values
            curr_true = v['true'].values
            result = self.calc_metric(torch.Tensor(curr_pred), torch.Tensor(curr_true), 'MAPE')
            hop_res[curr_hop] = result

        start_group = dict(list(res_data.groupby(by='start_hour')))
        start_res = {}
        for h, v in start_group.items():
            curr_hop = h
            curr_pred = v['pred'].values
            curr_true = v['true'].values
            result = self.calc_metric(torch.Tensor(curr_pred), torch.Tensor(curr_true), 'MAPE')
            start_res[curr_hop] = result

        plt.figure(figsize=(15, 4))
        plt.tight_layout()

        plt.subplot(121)
        plt.plot(hop_res.keys(), hop_res.values(), linewidth=2)
        plt.xlabel('Hop of the path')
        plt.ylabel('MAPE(%)')

        plt.subplot(122)
        plt.plot(hop_res.keys(), hop_res.values(), linewidth=2)
        plt.xlabel('Time of the day')
        plt.ylabel('MAPE(%)')

        plt.savefig(os.path.join(save_path, 'ETA.jpg'))

    def evaluate(self, save_path):
        """
        返回之前收集到的所有 batch 的评估结果
        """
        y_true = torch.cat(self.intermediate_result['true'], dim=0)
        y_pred = torch.cat(self.intermediate_result['pred'], dim=0)
        length = torch.cat(self.intermediate_result['length'], dim=0).squeeze()
        hop = torch.cat(self.intermediate_result['hop'], dim=0)
        start_hour = torch.cat(self.intermediate_result['start_hour'], dim=0)
        for metric in self.metrics:
            self.intermediate_result[metric].append(self.calc_metric(y_pred=y_pred, y_true=y_true, metric=metric))

        for metric in self.metrics:
            self.result[metric] = sum(self.intermediate_result[metric]) / \
                                  len(self.intermediate_result[metric])

        pickle.dump([y_true, y_pred, length, hop, start_hour, save_path],
                    open(os.path.join(save_path, 'full_result.pkl'), 'wb'))
        if self.config.get('plot_result', False):
            self.plot_result(y_true, y_pred, length, hop, start_hour, save_path)

        return self.result

    def save_result(self, save_path, filename=None):
        """
        将评估结果保存到 save_path 文件夹下的 filename 文件中

        Args:
            save_path: 保存路径
            filename: 保存文件名
        """
        self.evaluate(save_path)
        ensure_dir(save_path)
        if filename is None:  # 使用时间戳
            filename = str(self.config['exp_id']) + '_' + \
                       datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S') + '_' + \
                       self.config['model_name'] + '_' + self.config['dataset']

        if 'json' in self.save_modes:
            self._logger.info('Evaluate result is ' + json.dumps(self.result))
            path = os.path.join(save_path, '{}.json'.format(filename))
            with open(path, 'w') as f:
                json.dump(self.result, f)
            self._logger.info('Evaluate result is saved at ' + path)
            self._logger.info("\n" + json.dumps(self.result, indent=1))

        dataframe = {}
        if 'csv' in self.save_modes:
            for metric in self.metrics:
                dataframe[metric] = []
            for metric in self.metrics:
                dataframe[metric].append(self.result[metric])
            dataframe = pd.DataFrame(dataframe, index=range(1, 2))
            path = os.path.join(save_path, '{}.csv'.format(filename))
            dataframe.to_csv(path, index=False)
            self._logger.info('Evaluate result is saved at ' + path)
            self._logger.info("\n" + str(dataframe))
        return dataframe

    def clear(self):
        """
        清除之前收集到的 batch 的评估信息，适用于每次评估开始时进行一次清空，排除之前的评估输入的影响。
        """
        self.result = {}
        self.intermediate_result = {}
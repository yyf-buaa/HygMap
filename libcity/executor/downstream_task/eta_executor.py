import logging
import os
import time

import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from libcity.evaluator.downstream_task.regression_evaluator import RegressionEvaluator
from libcity.utils.downstream_utils import CosineLRScheduler
from libcity.utils import ensure_dir


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'linear.weight':
            return torch.sum(torch.square(param))


class AbstractExecutor(object):

    def __init__(self, args, model):
        self.config = vars(args)
        self.args = args
        self._logger = logging.getLogger()

        self.vocab_size = self.config.get('vocab_size')
        self.usr_num = self.config.get('usr_num')

        self.exp_id = self.config.get('exp_id', None)
        self.device = self.config.get('device', torch.device('cpu'))
        self.eta_epochs = self.config.get('max_epoch_eta', 100)
        self.sim_epochs = self.config.get('max_epoch_sim', 100)
        self.model_name = self.config.get('model_name', '')

        self.learner = self.config.get('learner', 'adamw')
        self.learning_rate = self.config.get('lr', 1e-4)
        self.weight_decay = self.config.get('weight_decay', 0.01)
        self.lr_beta1 = self.config.get('lr_beta1', 0.9)
        self.lr_beta2 = self.config.get('lr_beta2', 0.999)
        self.lr_betas = (self.lr_beta1, self.lr_beta2)
        self.lr_alpha = self.config.get('lr_alpha', 0.99)
        self.lr_epsilon = self.config.get('lr_epsilon', 1e-8)
        self.lr_momentum = self.config.get('lr_momentum', 0)
        self.grad_accmu_steps = self.config.get('grad_accmu_steps', 1)
        self.test_every = self.config.get('test_every', 10)

        self.lr_decay = self.config.get('lr_decay', False)
        self.lr_scheduler_type = self.config.get('lr_scheduler', 'cosinelr')
        self.lr_decay_ratio = self.config.get('lr_decay_ratio', 0.1)
        self.milestones = self.config.get('steps', [])
        self.step_size = self.config.get('step_size', 10)
        self.lr_lambda = self.config.get('lr_lambda', lambda x: x)
        self.lr_T_max = self.config.get('lr_T_max', 30)
        self.lr_eta_min = self.config.get('lr_eta_min', 0)
        self.lr_patience = self.config.get('lr_patience', 10)
        self.lr_threshold = self.config.get('lr_threshold', 1e-4)
        self.lr_warmup_epoch = self.config.get("lr_warmup_epoch", 5)
        self.lr_warmup_init = self.config.get("lr_warmup_init", 1e-6)
        self.t_in_epochs = self.config.get("t_in_epochs", True)

        self.clip_grad_norm = self.config.get('clip_grad_norm', False)
        self.max_grad_norm = self.config.get('max_grad_norm', 1.)
        self.use_early_stop = self.config.get('use_early_stop', False)
        self.patience = self.config.get('patience', 50)
        self.log_every = self.config.get('log_every', 1)
        self.log_batch = self.config.get('log_batch', 100)
        self.saved = self.config.get('saved_model', True)
        self.load_best_epoch = self.config.get('load_best_epoch', True)
        self.hyper_tune = self.config.get('hyper_tune', False)
        self.l2_reg = self.config.get('l2_reg', None)

        self.lape_dim = self.config.get('lape_dim', 8)
        self.random_flip = self.config.get('random_flip', True)
        self.add_lap = self.config.get('add_lap', True)

        self.cache_dir = './libcity/cache/{}/model_cache'.format(self.exp_id)
        self.png_dir = './libcity/cache/{}'.format(self.exp_id)
        self.evaluate_res_dir = './libcity/cache/{}/evaluate_cache'.format(self.exp_id)
        self.summary_writer_dir = './libcity/cache/{}'.format(self.exp_id)
        ensure_dir(self.cache_dir)
        ensure_dir(self.png_dir)
        ensure_dir(self.evaluate_res_dir)
        ensure_dir(self.summary_writer_dir)
        self._writer = SummaryWriter(self.summary_writer_dir)

        self.model = model.to(self.device)
        self._logger.info(self.model)
        for name, param in self.model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
        total_num = sum([param.nelement() for param in self.model.parameters()])
        self._logger.info('Total parameter numbers: {}'.format(total_num))
        if total_num>0:
            self.optimizer = self._build_optimizer()
            self.lr_scheduler = self._build_lr_scheduler()
            self.optimizer.zero_grad()
        # self.evaluator = get_evaluator(self.config, self.data_feature)

    def save_model(self, cache_name):
        """
        将当前的模型保存到文件

        Args:
            cache_name(str): 保存的文件名
        """
        ensure_dir(self.cache_dir)
        config = dict()
        config['model'] = self.model.cpu()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        torch.save(config, cache_name)
        self.model.to(self.device)
        self._logger.info("Saved model at " + cache_name)

    def load_model(self, cache_name):
        """
        加载对应模型的 cache

        Args:
            cache_name(str): 保存的文件名
        """
        assert os.path.exists(cache_name), 'Weights at {} not found' % cache_name
        checkpoint = torch.load(cache_name, map_location='cpu')
        self.model = checkpoint['model'].to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at " + cache_name)

    def save_model_with_epoch(self, epoch):
        """
        保存某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        ensure_dir(self.cache_dir)
        config = dict()
        config['model'] = self.model.cpu()
        config['optimizer_state_dict'] = self.optimizer.state_dict()
        config['epoch'] = epoch
        model_path = self.cache_dir + '/' + self.config['model_name'] + '_' + self.config[
            'dataset'] + '_epoch%d.tar' % epoch
        torch.save(config, model_path)
        self.model.to(self.device)
        self._logger.info("Saved model at {}".format(epoch))
        return model_path

    def load_model_with_epoch(self, epoch):
        """
        加载某个epoch的模型

        Args:
            epoch(int): 轮数
        """
        model_path = self.cache_dir + '/' + self.config['model_name'] + '_' + self.config[
            'dataset'] + '_epoch%d.tar' % epoch
        assert os.path.exists(model_path), 'Weights at epoch %d not found' % epoch
        checkpoint = torch.load(model_path, map_location='cpu')
        self.model = checkpoint['model'].to(self.device)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._logger.info("Loaded model at {}".format(epoch))

    def _build_optimizer(self):
        """
        根据全局参数`learner`选择optimizer
        """
        self._logger.info('You select `{}` optimizer.'.format(self.learner.lower()))
        if self.learner.lower() == 'adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate,
                                          eps=self.lr_epsilon, betas=self.lr_betas, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                        momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'adagrad':
            optimizer = torch.optim.Adagrad(self.model.parameters(), lr=self.learning_rate,
                                            eps=self.lr_epsilon, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'rmsprop':
            optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.learning_rate,
                                            alpha=self.lr_alpha, eps=self.lr_epsilon,
                                            momentum=self.lr_momentum, weight_decay=self.weight_decay)
        elif self.learner.lower() == 'sparse_adam':
            optimizer = torch.optim.SparseAdam(self.model.parameters(), lr=self.learning_rate,
                                               eps=self.lr_epsilon, betas=self.lr_betas)
        else:
            self._logger.warning('Received unrecognized optimizer, set default Adam optimizer')
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate,
                                         eps=self.lr_epsilon, weight_decay=self.weight_decay)
        return optimizer

    def _build_lr_scheduler(self):
        """
        根据全局参数`lr_scheduler`选择对应的lr_scheduler
        """
        if self.lr_decay:
            self._logger.info('You select `{}` lr_scheduler.'.format(self.lr_scheduler_type.lower()))
            if self.lr_scheduler_type.lower() == 'multisteplr':
                lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                    self.optimizer, milestones=self.milestones, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'steplr':
                lr_scheduler = torch.optim.lr_scheduler.StepLR(
                    self.optimizer, step_size=self.step_size, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'exponentiallr':
                lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                    self.optimizer, gamma=self.lr_decay_ratio)
            elif self.lr_scheduler_type.lower() == 'cosineannealinglr':
                lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    self.optimizer, T_max=self.lr_T_max, eta_min=self.lr_eta_min)
            elif self.lr_scheduler_type.lower() == 'lambdalr':
                lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optimizer, lr_lambda=self.lr_lambda)
            elif self.lr_scheduler_type.lower() == 'reducelronplateau':
                lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    self.optimizer, mode='min', patience=self.lr_patience,
                    factor=self.lr_decay_ratio, threshold=self.lr_threshold)
            elif self.lr_scheduler_type.lower() == 'cosinelr':
                lr_scheduler = CosineLRScheduler(
                    self.optimizer, t_initial=self.eta_epochs, lr_min=self.lr_eta_min, decay_rate=self.lr_decay_ratio,
                    warmup_t=self.lr_warmup_epoch, warmup_lr_init=self.lr_warmup_init, t_in_epochs=self.t_in_epochs)
            else:
                self._logger.warning('Received unrecognized lr_scheduler, '
                                     'please check the parameter `lr_scheduler`.')
                lr_scheduler = None
        else:
            lr_scheduler = None
        return lr_scheduler

    def train(self, train_dataloader, eval_dataloader, test_dataloader=None):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
            test_dataloader(torch.Dataloader): Dataloader
        """
        raise NotImplementedError("Executor train not implemented")

    def _train_epoch(self, train_dataloader, epoch_idx):
        raise NotImplementedError("Executor evaluate not implemented")

    def _valid_epoch(self, eval_dataloader, epoch_idx, mode='Eval'):
        raise NotImplementedError("Executor evaluate not implemented")

    def _draw_png(self, data):
        raise NotImplementedError("Executor evaluate not implemented")

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        raise NotImplementedError("Executor evaluate not implemented")


class ETAExecutor(AbstractExecutor):
    def __init__(self, args, model):
        super().__init__(args, model)
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.freeze = self.config.get("freeze", False)
        self.model = model
        model.to(self.device)
        self.evaluator = RegressionEvaluator(self.args)

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        # start_time = time.time()
        self._valid_epoch(test_dataloader, 0, mode='Test')
        # end_time = time.time()
        # self._logger.info('Test time {}s.'.format(end_time - start_time))

    def train(self, train_dataloader, eval_dataloader, test_dataloader=None):
        """
        use data to train model with config

        Args:
            train_dataloader(torch.Dataloader): Dataloader
            eval_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = -1
        train_time = []
        eval_time = []
        train_loss = []
        eval_loss = []
        lr_list = []

        num_batches = len(train_dataloader)
        self._logger.info("Num_batches: train={}, eval={}".format(num_batches, len(eval_dataloader)))

        for epoch_idx in range(self.eta_epochs):
            start_time = time.time()
            train_avg_loss = self._train_epoch(train_dataloader, epoch_idx)
            t1 = time.time()
            train_time.append(t1 - start_time)
            train_loss.append(train_avg_loss)
            self._logger.info("epoch complete!")

            self._logger.info("evaluating now!")
            t2 = time.time()
            eval_avg_loss = self._valid_epoch(eval_dataloader, epoch_idx, mode='Eval')
            end_time = time.time()
            eval_time.append(end_time - t2)
            eval_loss.append(eval_avg_loss)

            if self.lr_scheduler is not None:
                if self.lr_scheduler_type.lower() == 'reducelronplateau':
                    self.lr_scheduler.step(eval_avg_loss)
                elif self.lr_scheduler_type.lower() == 'cosinelr':
                    self.lr_scheduler.step(epoch_idx + 1)
                else:
                    self.lr_scheduler.step()

            log_lr = self.optimizer.param_groups[0]['lr']
            lr_list.append(log_lr)
            if (epoch_idx % self.log_every) == 0:
                message = 'Epoch [{}/{}] ({})  train_loss: {:.4f}, val_loss: {:.4f}, lr: {:.6f}, {:.2f}s'. \
                    format(epoch_idx, self.eta_epochs, (epoch_idx + 1) * num_batches, train_avg_loss,
                           eval_avg_loss, log_lr, (end_time - start_time))
                self._logger.info(message)

            if eval_avg_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f}, '
                                      'saving to {}'.format(min_val_loss, eval_avg_loss, model_file_name))
                min_val_loss = eval_avg_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break

            if (epoch_idx + 1) % self.test_every == 0:
                self.evaluate(test_dataloader)

        if len(train_time) > 0:
            self._logger.info('Trained totally {} epochs, average train time is {:.3f}s, '
                              'average eval time is {:.3f}s'.
                              format(len(train_time), sum(train_time) / len(train_time),
                                     sum(eval_time) / len(eval_time)))
        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)

        return min_val_loss

    def _train_epoch(self, train_dataloader, epoch_idx):
        """
        完成模型一个轮次的训练

        Args:
            train_dataloader: 训练数据
            epoch_idx: 轮次数

        Returns:
            list: 每个batch的损失的数组
        """
        batches_seen = epoch_idx * len(train_dataloader)  # 总batch数

        self.model = self.model.train()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total masked elements in epoch

        for i, data in tqdm(enumerate(train_dataloader), desc="Train epoch={}".format(
                epoch_idx), total=len(train_dataloader)):

            data = {key: value.to(self.device) for key, value in data.items()}
            targets = data["eta_target"]
            targets = targets.reshape((targets.shape[0], 1))

            predictions = self.model(data["road_emb"], data["input_mask"])

            batch_loss_list = self.criterion(predictions, targets)
            batch_loss = torch.sum(batch_loss_list)
            num_active = len(batch_loss_list)  # batch_size
            mean_loss = batch_loss / num_active  # mean loss (over samples) used for optimization
            if self.l2_reg is not None:
                total_loss = mean_loss + self.l2_reg * l2_reg_loss(self.model)
            else:
                total_loss = mean_loss

            total_loss = total_loss / self.grad_accmu_steps
            batches_seen += 1
            # with torch.autograd.detect_anomaly():
            total_loss.backward()

            if self.clip_grad_norm:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            if batches_seen % self.grad_accmu_steps == 0:
                self.optimizer.step()
                if self.lr_scheduler_type == 'cosinelr' and self.lr_scheduler is not None:
                    self.lr_scheduler.step_update(num_updates=batches_seen // self.grad_accmu_steps)
                self.optimizer.zero_grad()

            with torch.no_grad():
                total_active_elements += num_active
                epoch_loss += batch_loss.item()  # add total loss of batch
            post_fix = {
                "mode": "Train",
                "epoch": epoch_idx,
                "iter": i,
                "lr": self.optimizer.param_groups[0]['lr'],
                "loss": mean_loss.item(),
            }
            if i % self.log_batch == 0:
                self._logger.info(str(post_fix))
        epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
        self._logger.info("Train: expid = {}, Epoch = {}, avg_loss = {}.".format(
            self.exp_id, epoch_idx, epoch_loss))
        self._writer.add_scalar('Train loss', epoch_loss, epoch_idx)
        return epoch_loss

    def _valid_epoch(self, eval_dataloader, epoch_idx, mode='Eval'):
        """
        完成模型一个轮次的评估

        Args:
            eval_dataloader: 评估数据
            epoch_idx: 轮次数

        Returns:
            float: 评估数据的平均损失值
        """
        self.model = self.model.eval()
        if mode == 'Test':
            self.evaluator.clear()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total masked elements in epoch

        labels = []
        preds = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(eval_dataloader), desc="{} epoch={}".format(
                    mode, epoch_idx), total=len(eval_dataloader)):
                data = {key: value.to(self.device) for key, value in data.items()}
                targets = data["eta_target"]
                targets = targets.reshape((targets.shape[0], 1))
                hop = data['hop']
                length = data['length']
                tlist = data['time_list']
                #hop_list = [item[0] for item in data['hop']]
                predictions = self.model(data["road_emb"], data["input_mask"])
                if mode == 'Test':
                    preds.append(predictions.cpu().numpy())
                    labels.append(targets.cpu().numpy())
                    evaluate_input = {
                        'y_true': targets,
                        'y_pred': predictions,
                        'hop': hop,
                        'length': length,
                        'tlist': tlist
                    }
                    self.evaluator.collect(evaluate_input)

                batch_loss_list = self.criterion(predictions, targets)
                batch_loss = torch.sum(batch_loss_list)
                num_active = len(batch_loss_list)  # batch_size
                mean_loss = batch_loss / num_active  # mean loss (over samples) used for optimization

                total_active_elements += num_active
                epoch_loss += batch_loss.item()  # add total loss of batch

                post_fix = {
                    "mode": mode,
                    "epoch": epoch_idx,
                    "iter": i,
                    "lr": self.optimizer.param_groups[0]['lr'],
                    "loss": mean_loss.item(),
                }
                if i % self.log_batch == 0:
                    self._logger.info(str(post_fix))

            epoch_loss = epoch_loss / total_active_elements  # average loss per element for whole epoch
            self._logger.info("{}: expid = {}, Epoch = {}, avg_loss = {}.".format(
                mode, self.exp_id, epoch_idx, epoch_loss))
            self._writer.add_scalar('{} loss'.format(mode), epoch_loss, epoch_idx)

            if mode == 'Test':
                preds = np.concatenate(preds, axis=0)
                labels = np.concatenate(labels, axis=0)
                np.save(self.cache_dir + '/eta_labels.npy', labels)
                np.save(self.cache_dir + '/eta_preds.npy', preds)
                self.evaluator.save_result(self.evaluate_res_dir)
            return epoch_loss
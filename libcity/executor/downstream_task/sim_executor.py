import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from libcity.executor.downstream_task.eta_executor import AbstractExecutor
import torch


def l2_reg_loss(model):
    """Returns the squared L2 norm of output layer of given model"""

    for name, param in model.named_parameters():
        if name == 'linear.weight':
            return torch.sum(torch.square(param))


class SimExecutor(AbstractExecutor):
    def __init__(self, args, model,train_sim_label,val_sim_label,test_sim_label):
        super().__init__(args, model)
        self.base_path = args.cache_dir
        self.evaluate_res_csv_path = self.base_path + '/evaluate_cache/{}_sim_evaluate_res_{}.csv' \
            .format(self.exp_id, args.dataset)
        self.evaluate_simi_pred_path = self.base_path + '/evaluate_cache/simi_pred.npy'
        print(self.evaluate_res_csv_path)
        self.criterion = torch.nn.MSELoss(reduction='none')
        self.freeze = self.config.get("freeze", False)
        self.model = model
        self.train_sim_label =train_sim_label
        self.val_sim_label = val_sim_label
        self.test_sim_label = test_sim_label
        model.to(self.device)

    def evaluate(self, test_dataloader):
        """
        use model to test data

        Args:
            test_dataloader(torch.Dataloader): Dataloader
        """
        self._logger.info('Start evaluating ...')
        #epoch_loss,sim_preds = self._test_epoch(test_dataloader, 0, mode='Test')
        sim_preds = None
        sim_target = None
        batch_loss = None
        for i, data in tqdm(enumerate(test_dataloader)):
            data = {key: value.to(self.device) for key, value in data.items()}
            traj_embs = self.model(data["road_emb"], data["input_mask"])
            sim_preds = torch.cdist(traj_embs, traj_embs, 1).cpu().detach().numpy()
            sim_target = self.test_sim_label
            pred_l1_simi = torch.cdist(traj_embs, traj_embs, 1)
            pred_l1_simi = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal=1) == 1].float()
            sub_simi = torch.from_numpy(self.test_sim_label).to(self.device)
            truth_l1_simi = sub_simi[torch.triu(torch.ones(sub_simi.shape), diagonal=1) == 1].float()
            batch_loss_list = self.criterion(pred_l1_simi, truth_l1_simi)
            batch_loss = torch.sum(batch_loss_list)
        self._logger.info('test_simi_preds is '+str(sim_preds.shape))
        self._logger.info('test_simi_traget is ' + str(sim_target.shape))
        self._logger.info('test_loss = '+str(batch_loss))
        np.save(self.evaluate_simi_pred_path,sim_preds)
        self.evaluate_most_sim(sim_target,sim_preds)


    def evaluate_most_sim(self,sim_label,sim_preds):
        #给sim_label和sim_pred的对角元素都变成inf，不然label全成自己了
        np.fill_diagonal(sim_preds,np.inf)
        np.fill_diagonal(sim_label,np.inf)
        #给sim_preds每一行排序
        topk = [1, 5, 10,20]
        sorted_pred_index = sim_preds.argsort(axis=1)
        label_most_sim = np.argmin(sim_label,axis=1)
        total_num = sim_preds.shape[0]
        self.result = {}
        hit = {}
        for k in topk:
            hit[k] = 0
        rank = 0
        rank_p = 0.0
        for i in tqdm(range(total_num)):
            # 在sim_labe找到这一行distance最小的的作为真正的最相似轨迹
            label = label_most_sim[i]
            rank_list = list(sorted_pred_index[i])
            rank_index = rank_list.index(label)
            rank += (rank_index + 1)
            rank_p += 1.0 / (rank_index + 1)
            for k in topk:
                if label in sorted_pred_index[i][:k]:
                    hit[k] += 1
        self.result['MR'] = rank / total_num
        self.result['MRR'] = rank_p / total_num
        for k in topk:
            self.result['HR@{}'.format(k)] = hit[k] / total_num
        self._logger.info("Evaluate result is {}".format(self.result))
        dataframe = pd.DataFrame(self.result, index=range(1, 2))
        dataframe.to_csv(self.evaluate_res_csv_path, index=False)
        self._logger.info('Evaluate result is saved at {}'.format(self.evaluate_res_csv_path))

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

        for epoch_idx in range(self.sim_epochs):
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
                    format(epoch_idx, self.sim_epochs, (epoch_idx + 1) * num_batches, train_avg_loss,
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
            # targets = data["sim_label"]
            # targets = targets.reshape((targets.shape[0], 1))

            traj_embs = self.model(data["road_emb"], data["input_mask"])
            pred_l1_simi = torch.cdist(traj_embs, traj_embs, 1)
            pred_l1_simi = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal=1) == 1].float()
            batch_indexes = data['index'].tolist()
            sub_simi = torch.from_numpy(self.train_sim_label[batch_indexes][:, batch_indexes]).to(self.device)
            truth_l1_simi = sub_simi[torch.triu(torch.ones(sub_simi.shape), diagonal=1) == 1].float()
            batch_loss_list = self.criterion(pred_l1_simi, truth_l1_simi)
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
        # if mode == 'Test':
        #     self.evaluator.clear()

        epoch_loss = 0  # total loss of epoch
        total_active_elements = 0  # total masked elements in epoch

        labels = []
        preds = []

        with torch.no_grad():
            for i, data in tqdm(enumerate(eval_dataloader), desc="{} epoch={}".format(
                    mode, epoch_idx), total=len(eval_dataloader)):
                data = {key: value.to(self.device) for key, value in data.items()}
                traj_embs = self.model(data["road_emb"], data["input_mask"])
                pred_l1_simi = torch.cdist(traj_embs, traj_embs, 1)
                pred_l1_simi = pred_l1_simi[torch.triu(torch.ones(pred_l1_simi.shape), diagonal=1) == 1].float()
                batch_indexes = data['index'].tolist()
                sub_simi = torch.from_numpy(self.val_sim_label[batch_indexes][:, batch_indexes]).to(self.device)
                truth_l1_simi = sub_simi[torch.triu(torch.ones(sub_simi.shape), diagonal=1) == 1].float()
                batch_loss_list = self.criterion(pred_l1_simi, truth_l1_simi)
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

            return epoch_loss

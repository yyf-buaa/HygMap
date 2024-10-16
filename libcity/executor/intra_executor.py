import time
import numpy as np
from logging import getLogger
from libcity.utils import get_evaluator
from libcity.executor.abstract_executor import AbstractExecutor
from libcity.model.utils import *


class IntraExecutor(AbstractExecutor):
    def __init__(self, config, model, data_feature):
        self._logger = getLogger()
        #self.evaluator = get_evaluator(config, data_feature)
        self.config = config
        self.exp_id = self.config.get('exp_id', None)
        self.device = self.config.get('device', torch.device('cpu'))
        self.epochs = self.config.get('max_epoch', 100)
        self.model_name = self.config.get('model', '')
        self.dataset = self.config.get('dataset', '')
        self.seed = self.config.get('seed', '')

        super().__init__(config, model, data_feature)
        self.output_dim = config['output_dim']
        self.cache_file = './libcity/cache/{}/evaluate_cache/embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model_name, self.dataset, self.output_dim)

    def get_emb(self, data):
        z =  self.model.encode_poi_road_region(data_x=data['node'].x,graph_edge_index=data.edge_index_dict,emb=True)
        emb_to_save = z.cpu().detach().numpy()
        return emb_to_save

    def evaluate(self, data):
        self.model.eval()
        emb_to_save = self.get_emb(data)
        np.save(self.cache_file, emb_to_save)
        # self.evaluator.clear()
        # self.evaluator.evaluate()

    def train(self, data, _, __=None):
        self._logger.info('Start training ...')
        min_val_loss = float('inf')
        wait = 0
        best_epoch = -1
        train_time = []
        eval_time = []

        for epoch_idx in range(self.epochs):
            avg_loss, ti = self._train_epoch(data, epoch_idx)
            train_time.append(ti)

            t2 = time.time()
            end_time = time.time()
            eval_time.append(end_time - t2)
            self._logger.info('Val Epoch {}/{} complete, avg_loss={:4f}, time={:3f}s'.format(epoch_idx, self.epochs, avg_loss, end_time - t2))

            if avg_loss < min_val_loss:
                wait = 0
                if self.saved:
                    model_file_name = self.save_model_with_epoch(epoch_idx)
                    self._logger.info('Val loss decrease from {:.4f} to {:.4f},'
                                      'saving to {}'.format(min_val_loss, avg_loss, model_file_name))
                min_val_loss = avg_loss
                best_epoch = epoch_idx
            else:
                wait += 1
                if wait == self.patience and self.use_early_stop:
                    self._logger.warning('Early stopping at epoch: %d' % epoch_idx)
                    break

        if self.load_best_epoch:
            self.load_model_with_epoch(best_epoch)

        return min_val_loss

    def _train_epoch(self, data, epoch_idx):
        start_time = time.time()

        self.model.train()
        self.optimizer.zero_grad()

        z1 , z2 , z = self.model(data)
        loss = self.model.intra_loss(z1, z2, batch_size=512)
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        train_time = time.time() - start_time
        self._logger.info('Epoch {} complete,'
                          'loss={:6f}, time={:3f}s'.format(epoch_idx, loss.item(), train_time))
        return loss.item(), train_time

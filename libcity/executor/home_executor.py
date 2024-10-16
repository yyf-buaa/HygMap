import time
import numpy as np
from logging import getLogger
from libcity.utils import get_evaluator
from libcity.executor.abstract_executor import AbstractExecutor
from libcity.model.utils import *


class HOMEExecutor(AbstractExecutor):
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
        self.road_cl_ratio = config['road_cl_ratio']
        self.region_cl_ratio = config['region_cl_ratio']
        self.poi_cl_ratio = config['poi_cl_ratio']
        self.mask_fill_ratio = config['mask_fill_ratio']
        self.form_region_ratio = config['form_region_ratio']
        self.output_dim = config['output_dim']
        self.poi_cache_file = './libcity/cache/{}/evaluate_cache/poi_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model_name, self.dataset, self.output_dim)
        self.road_cache_file = './libcity/cache/{}/evaluate_cache/road_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model_name, self.dataset, self.output_dim)
        self.region_cache_file = './libcity/cache/{}/evaluate_cache/region_embedding_{}_{}_{}.npy'. \
            format(self.exp_id, self.model_name, self.dataset, self.output_dim)

    def get_emb(self, data):
        poi_z,road_z,region_z =  self.model.encode_poi_road_region(data_poi_x=data['intra_graph']['poi_node'].x, data_road_x=data['intra_graph']['road_node'].x,data_region_x= data['intra_graph']['region_node'].x,
                                                                         graph_edge_index=data['intra_graph'].edge_index_dict,
                                                                         hyperedge_index=data['hyperedge_index'],emb=True)
        road_emb_to_save = road_z.cpu().detach().numpy()
        region_emb_to_save = region_z.cpu().detach().numpy()
        poi_emb_to_save = poi_z.cpu().detach().numpy()
        return poi_emb_to_save, road_emb_to_save, region_emb_to_save

    def evaluate(self, data):
        self.model.eval()
        poi_emb_to_save, road_emb_to_save, region_emb_to_save = self.get_emb(data)
        np.save(self.poi_cache_file, poi_emb_to_save)
        np.save(self.road_cache_file, road_emb_to_save)
        np.save(self.region_cache_file, region_emb_to_save)
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

        poi_z1 , road_z1, region_z1, poi_z2 , road_z2, region_z2, poi_z, road_z, region_z = self.model(data)
        poi_loss = self.model.poi_loss(poi_z1, poi_z2, batch_size=512)
        road_loss = self.model.road_loss(road_z1, road_z2, batch_size=512)
        region_loss = self.model.region_loss(region_z1, region_z2, None)
        mask_fill_loss = self.model.mask_fill_loss(poi_z, road_z, self.model.set_project_head1.to('cuda:1'))
        form_region_loss = self.model.form_region_loss(poi_z, road_z, region_z, self.model.set_project_head2.to('cuda:1'))
        loss = self.road_cl_ratio * road_loss + self.region_cl_ratio * region_loss + self.poi_cl_ratio * poi_loss + self.mask_fill_ratio*mask_fill_loss + self.form_region_ratio*form_region_loss
        loss.backward()
        if self.clip_grad_norm:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        train_time = time.time() - start_time
        self._logger.info('Epoch {}/{} complete, poi_cl_loss = {:6f}, road_cl_loss={:6f}, region_cl_loss={:6f}, '
                          ' mask_fill_loss = {:6f}, form_region_loss = {:6f},'
                          'loss={:6f}, time={:3f}s'.format(epoch_idx, self.epochs,poi_loss.item(),road_loss.item(),
                                                          region_loss.item(),mask_fill_loss.item(),form_region_loss.item(), loss.item(), train_time))
        return loss.item(), train_time

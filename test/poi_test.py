from logging import getLogger
from sklearn.metrics import accuracy_score, f1_score
import json
import pandas as pd
from sklearn import linear_model
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_absolute_percentage_error
import torch
import torch.nn as nn
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn import manifold
import seaborn as sns
import random
def ensure_dir(dir_path):
    """Make sure the directory exists, if it does not exist, create it.

    Args:
        dir_path (str): directory path
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def gen_index_map(df, column, offset=0):
    index_map = {origin: index + offset
                 for index, origin in enumerate(df[column].drop_duplicates())}
    return index_map

class Classifier(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class Bilinear_Module(nn.Module):
    def __init__(self, dim):
        super(Bilinear_Module, self).__init__()
        self.regressor = nn.Bilinear(dim, dim, 1)

    def forward(self, x):
        # x为batch_size×2×dim
        return self.regressor(x[:, 0, :], x[:, 1, :])


def metrics_local(y_truths, y_preds):
    y_preds[y_preds < 0] = 0
    mae = mean_absolute_error(y_truths, y_preds)
    rmse = mean_squared_error(y_truths, y_preds, squared=False)
    mape = mean_absolute_percentage_error(y_truths, y_preds) * 100
    r2 = r2_score(y_truths, y_preds)
    return mae, rmse, mape, r2


def evaluation_classify(X, y, kfold=5, num_classes=5, seed=42, output_dim=128):
    KF = StratifiedKFold(n_splits=kfold, random_state=seed, shuffle=True)
    y_preds = []
    y_trues = []
    for fold_num, (train_idx, val_idx) in enumerate(KF.split(X, y)):
        X_train, X_eval = X[train_idx], X[val_idx]
        y_train, y_eval = y[train_idx], y[val_idx]
        X_train = torch.tensor(X_train).cuda()
        X_eval = torch.tensor(X_eval).cuda()
        y_train = torch.tensor(y_train).cuda()
        y_eval = torch.tensor(y_eval).cuda()

        model = Classifier(output_dim, num_classes=num_classes).cuda()
        opt = torch.optim.Adam(model.parameters())

        best_acc = 0.
        best_pred = 0.
        for e in range(1000):
            model.train()
            opt.zero_grad()
            ce_loss = nn.CrossEntropyLoss()(model(X_train), y_train)
            ce_loss.backward()
            opt.step()

            model.eval()
            y_pred = torch.argmax(model(X_eval), -1).detach().cpu()
            acc = accuracy_score(y_eval.cpu(), y_pred, normalize=True)
            if acc > best_acc:
                best_acc = acc
                best_pred = y_pred
        y_preds.append(best_pred)
        y_trues.append(y_eval.cpu())

    y_preds = torch.cat(y_preds, dim=0)
    y_trues = torch.cat(y_trues, dim=0)
    macro_f1 = f1_score(y_trues, y_preds, average='macro')
    micro_f1 = f1_score(y_trues, y_preds, average='micro')
    return micro_f1, macro_f1


def evaluation_bilinear_reg(embedding, flow, kfold=5, seed=42, output_dim=128):
    kf = KFold(n_splits=kfold, random_state=seed, shuffle=True)
    X = []
    y = []
    node_num = embedding.shape[0]
    for i in range(node_num):
        for j in range(node_num):
            if flow[i][j] > 0:
                X.append([embedding[i], embedding[j]])
                y.append(flow[i][j])
    y_preds = []
    y_trues = []
    X = np.array(X)
    y = np.array(y)
    for fold_num, (train_idx, val_idx) in enumerate(kf.split(X, y)):
        X_train, X_eval = X[train_idx], X[val_idx]
        y_train, y_eval = y[train_idx], y[val_idx]
        X_train = torch.tensor(X_train).cuda()
        X_eval = torch.tensor(X_eval).cuda()
        y_train = torch.tensor(y_train).cuda()
        y_eval = torch.tensor(y_eval).cuda()
        model = Bilinear_Module(output_dim).cuda()
        criterion = nn.MSELoss()
        opt = torch.optim.Adam(model.parameters(), lr=0.01)
        best_mse = float('inf')
        best_pred = 0
        for e in range(2000):
            model.train()
            opt.zero_grad()
            mse_loss = criterion(model(X_train).squeeze(), y_train)
            mse_loss.backward()
            opt.step()
            model.eval()
            y_val_pred = model(X_eval).squeeze()
            val_loss = criterion(y_eval, y_val_pred)
            if val_loss < best_mse:
                best_mse = val_loss
                best_pred = y_val_pred
        y_preds.append(best_pred.detach().cpu())
        y_trues.append(y_eval.cpu())
    y_preds = torch.cat(y_preds, dim=0).cpu()
    y_trues = torch.cat(y_trues, dim=0).cpu()
    y_preds = y_preds.numpy()
    y_trues = y_trues.numpy()
    mae, rmse, mape, r2 = metrics_local(y_trues, y_preds)
    return mae, rmse


def evaluation_reg(embedding, label , kfold=5, seed=42, output_dim=128):
    X = []
    y = []
    kf = KFold(n_splits=kfold, random_state=seed, shuffle=True)
    node_num = embedding.shape[0]
    for i in range(node_num):
            if label[i] > 0:
                X.append(embedding[i])
                y.append(label[i])
    y_preds = []
    y_truths = []
    X = np.array(X)
    y = np.array(y)
    for fold_num, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        reg = linear_model.Ridge(alpha=1.0, random_state=seed)
        reg.fit(X_train, y_train)
        y_pred = reg.predict(X_test)

        y_preds.append(y_pred)
        y_truths.append(y_test)

    y_preds = np.concatenate(y_preds)
    y_truths = np.concatenate(y_truths)

    mae, rmse, mape, r2 = metrics_local(y_truths, y_preds)
    return mae, rmse

class POIEvaluator():

    def __init__(self,config):
        self._logger = getLogger('evaluator')
        self.result = {}
        self.model = config.get('model', '')
        self.dataset = config.get('dataset', '')
        self.cluster_kinds = config.get('cluster_kinds', 5)
        self.seed = config.get('seed', 0)
        self.data_path = './raw_data/' + self.dataset + '/'
        self.output_dim = config.get('output_dim', 128)
        self.poi_embedding_path = config.get('path', None)

    def collect(self, batch):
        pass

    def evaluation_cluster(self, y_truth, useful_index, node_emb, item_type):
        # kinds = self.cluster_kinds
        # self._logger.info('Start Kmeans, data.shape = {}, kinds = {}'.format(
        #     str(node_emb.shape), kinds))
        # k_means = KMeans(n_clusters=self.cluster_kinds, random_state=self.seed)
        # k_means.fit(node_emb)
        # labels = k_means.labels_
        # y_predict = k_means.predict(node_emb)
        # y_predict_useful = y_predict[useful_index]
        # nmi = normalized_mutual_info_score(y_truth, y_predict_useful)
        # ars = adjusted_rand_score(y_truth, y_predict_useful)
        # # SC指数
        # sc = float(metrics.silhouette_score(node_emb, labels, metric='euclidean'))
        # # DB指数
        # db = float(metrics.davies_bouldin_score(node_emb, labels))
        # # CH指数
        # ch = float(metrics.calinski_harabasz_score(node_emb, labels))
        # self._logger.info("Evaluate result is sc = {:6f}, db = {:6f}, ch = {:6f}, nmi = {:6f}, ars = {:6f}".format(sc, db, ch, nmi, ars))
        # result_path = './libcity/cache/{}/evaluate_cache/kmeans_evaluate_{}_{}_{}_{}_{}.json'. \
        #     format(self.exp_id, item_type, self.model, self.dataset, str(self.output_dim), str(kinds))
        # json.dump(self.result, open(result_path, 'w'), indent=4)
        # self._logger.info('Evaluate result is saved at {}'.format(result_path))

        # TSNE可视化
        plt.figure()
        x_input_tsne = normalize(node_emb, norm='l2', axis=1)  # 按行归一化
        tsne = manifold.TSNE(n_components=2, init='pca', random_state=self.seed)  # n_components=2降维为2维并且可视化
        x_tsne = tsne.fit_transform(x_input_tsne)
        sns.scatterplot(x=x_tsne[:, 0][useful_index], y=x_tsne[:, 1][useful_index], hue=y_truth, palette='Set1', legend=True, linewidth=0, )
        result_path = './test/result/{}_{}_{}_poi/evaluate_{}_{}_{}.png'. \
            format(self.model, self.dataset, str(self.output_dim), self.model, self.dataset, str(self.output_dim))
        plt.title('{} {} {}'.format(item_type, self.model, self.dataset))
        plt.savefig(result_path)
        # self._logger.info('Kmeans result for TSNE is saved at {}'.format(result_path))
        # return sc, db, ch, nmi, ars

    def _valid_clf(self, emb):
        data = pd.read_csv(
            './raw_data/{}/poimap_{}/poimap_{}.geo'.format(
                self.dataset, self.dataset, self.dataset))
        label_name = col_name = 'type'
        try:
            label = data[col_name].dropna().astype(int).values
        except:
            mp = gen_index_map(data, col_name)
            label = data[col_name].dropna().map(mp).values
        num_classes = 5
        num_classes = min(num_classes, len(set(label)))
        tmp = []
        for i in range(label.min(), label.max() + 1):
            tmp.append((label[label == i].shape[0], i))
        assert num_classes <= len(tmp)
        tmp.sort()
        tmp = [item[1] for item in tmp]
        useful_label = tmp[::-1][:num_classes]
        relabel = {}
        for i, j in enumerate(useful_label):
            relabel[j] = i
        useful_index = []
        self._logger.info(f'poi emb shape = {emb.shape}, label shape = {label.shape}')
        assert len(label) == len(emb)

        X = []
        y = []
        for i, label_i in enumerate(label):
            if label_i in useful_label:
                useful_index.append(i)
                X.append(emb[i: i + 1, :])
                y.append(relabel[label_i])
        X = np.concatenate(X, axis=0)
        y = np.array(y)

        self._logger.info(
            f'Selected poi emb shape = {X.shape}, label shape = {y.shape}, label min = {y.min()}, label max = {y.max()}, num_classes = {num_classes}')
        micro_f1, macro_f1 = evaluation_classify(X, y, kfold=5, num_classes=num_classes, seed=self.seed,
                                                 output_dim=self.output_dim)
        print('micro F1: {:6f}, macro F1: {:6f}'.format(micro_f1, macro_f1))
        return y, useful_index, micro_f1, macro_f1


    def _valid_poi_flow(self, poi_emb):
        self._logger.warning('Evaluating POI Flow Prediction')
        flow = np.load('./raw_data/{}/poi_check_in.npy'.format(
            self.dataset)).astype('float32')
        poi_mae, poi_rmse = evaluation_reg(
            poi_emb, flow, kfold=5, seed=self.seed, output_dim=self.output_dim)
        print(
            "Result of {} estimation in {}:".format(
                'flow', self.dataset))
        print('MAE = {:6f}, RMSE = {:6f}'.format(
            poi_mae, poi_rmse))
        return poi_mae, poi_rmse


    def evaluate_poi_embedding(self):
        poi_emb = np.load(self.poi_embedding_path)
        print(
            'Load poi emb {}, shape = {}'.format(
                self.poi_embedding_path,
                poi_emb.shape))

        self._logger.warning('Evaluating POI Classification')
        y_truth, useful_index, poi_micro_f1, poi_macro_f1 = self._valid_clf(
            poi_emb)
        self.evaluation_cluster(y_truth,useful_index,poi_emb,'poi')
        #poi_mae, poi_rmse = self._valid_poi_flow(poi_emb)
        #poi_bilinear_mae, poi_bilinear_rmse = self._valid_poi_flow_using_bilinear(poi_emb)
        #
        # self.result['poi_micro_f1'] = [poi_micro_f1]
        # self.result['poi_macro_f1'] = [poi_macro_f1]
        # self.result['poi_mae'] = [float(poi_mae)]
        # self.result['poi_rmse'] = [float(poi_rmse)]
        # self.result['poi_od_mae'] = [float(poi_bilinear_mae)]
        # self.result['poi_od_rmse'] = [float(poi_bilinear_rmse)]

    def evaluate(self):
        # self.evaluate_poi_embedding()
        ensure_dir('./test/result/{}_{}_{}_poi/'.format(self.model, self.dataset, str(self.output_dim), self.model))
        self.evaluate_poi_embedding()
        # ensure_dir('./test/result/{}_{}_{}_poi/'.format(self.model, self.dataset, str(self.output_dim),self.model))
        # result_path = './test/result/{}_{}_{}_poi/evaluate_{}_{}_{}.json'. \
        #     format(self.model, self.dataset, str(self.output_dim),self.model, self.dataset, str(self.output_dim))
        # print(self.result)
        # json.dump(self.result, open(result_path, 'w'), indent=4)
        # print('Evaluate result is saved at {}'.format(result_path))
        #
        # df = pd.DataFrame.from_dict(self.result, orient='columns')
        # print(df)
        # result_path = './test/result/{}_{}_{}_poi/evaluate_{}_{}_{}.csv'. \
        #     format(self.model, self.dataset, str(self.output_dim),self.model, self.dataset, str(self.output_dim))
        # df.to_csv(result_path, index=False)
        # print('Evaluate result is saved at {}'.format(result_path))
        return self.result

    def save_result(self, save_path, filename=None):
        pass

    def clear(self):
        self.result = {}

parser = argparse.ArgumentParser()
# 添加参数
parser.add_argument('--model', type=str)
parser.add_argument('--dataset', type=str)
parser.add_argument('--path', type=str)
# 解析参数
args = parser.parse_args()
args_dict = vars(args)
poi_evaluator = POIEvaluator(args_dict)
poi_evaluator.evaluate()
import numpy as np
import torch


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

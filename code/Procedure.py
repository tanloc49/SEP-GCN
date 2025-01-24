'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation
@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Design training and test process
'''
import world
import numpy as np
import torch
import utils
import metrics
from utils import timer
import model
import multiprocessing

CORES = multiprocessing.cpu_count() // 2


def BPR_train_original(dataset, recommend_model, loss_class, epoch, neg_k=1, w=None, base_seed=42):
    Recmodel = recommend_model
    Recmodel.train()
    bpr: utils.BPRLoss = loss_class

    epoch_seed = base_seed + epoch
    utils.set_seed(epoch_seed)

    with timer(name="Sample"):
        S = utils.UniformSample_original(dataset)
    users = torch.Tensor(S[:, 0]).long()
    posItems = torch.Tensor(S[:, 1]).long()
    negItems = torch.Tensor(S[:, 2]).long()

    users = users.to(world.device)
    posItems = posItems.to(world.device)
    negItems = negItems.to(world.device)
    users, posItems, negItems = utils.shuffle(users, posItems, negItems)

    total_batch = len(users) // world.config['bpr_batch_size'] + 1
    aver_loss = 0.

    for batch_i, (batch_users, batch_pos, batch_neg) in enumerate(
            utils.minibatch(users, posItems, negItems, batch_size=world.config['bpr_batch_size'])):

        batch_seed = epoch_seed + batch_i
        utils.set_seed(batch_seed)

        cri = bpr.stageOne(batch_users, batch_pos, batch_neg, epoch)
        aver_loss += cri

        if world.tensorboard:
            w.add_scalar(f'BPRLoss/BPR', cri, epoch * int(len(users) / world.config['bpr_batch_size']) + batch_i)

    aver_loss = aver_loss / total_batch
    time_info = timer.dict()
    timer.zero()
    return f"loss: {aver_loss:.5f}", aver_loss
    
    
def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    pre, recall, ndcg, acc, mrr = [], [], [], [], []
    for k in world.topks:
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        ndcg.append(utils.NDCGatK_r(groundTrue, r, k))
        acc.append(utils.accuracy_at_k(groundTrue, sorted_items, k))

    mrr_value = utils.mean_reciprocal_rank(groundTrue, sorted_items)
    return {
        'recall': np.array(recall),
        'precision': np.array(pre),
        'ndcg': np.array(ndcg),
        'accuracy': np.array(acc),
        'mrr': mrr_value
    }
        
            
def Test(dataset, Recmodel, epoch, w=None, multicore=0):
    dataset: utils.BasicDataset
    testDict: dict = dataset.testDict
    Recmodel: model.SEPGCN

    Recmodel = Recmodel.eval()
    max_K = max(world.topks)

    with torch.no_grad():
        users = list(testDict.keys())
        groundTrue = [testDict[u] for u in users]
        users_gpu = torch.Tensor(users).long().to(world.device)

        rating = Recmodel.getUsersRating(users_gpu)
        allPos = dataset.getUserPosItems(users_gpu)
        exclude_index = []
        exclude_items = []
        for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i] * len(items))
            exclude_items.extend(items)

        rating[exclude_index, exclude_items] = -(1 << 10)
        _, rating_K = torch.topk(rating, k=max_K)
        rating_K = rating_K.cpu().numpy()

        results = metrics.evaluate_metrics(rating_K, groundTrue, world.topks, epoch)

        del rating
        return results

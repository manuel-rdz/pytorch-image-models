""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    print('-----------out/pred shapes---------')
    print(output.shape)
    print(target.shape)
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]


def auc_score(y_true, y_pred):
    if len(y_true.shape) == 1:
        score = roc_auc_score(y_true, y_pred)
        return score, [score]

    scores = []
    for i in range(y_true.shape[1]):
        try:
            score = roc_auc_score(y_true[:,i], y_pred[:,i])
            scores.append(score)
        except ValueError:
            pass
    avg_score = np.mean(scores)
    return avg_score, scores

def mAP_score(y_true, y_pred):
    if len(y_true.shape) == 1:
        score = average_precision_score(y_true, y_pred)
        return score, [score]
    
    scores = []
    for i in range(y_true.shape[1]):
        try:
            score = average_precision_score(y_true[:, i], y_pred[:, i])
            scores.append(score)
        except ValueError:
            pass
    avg_score = np.mean(scores)
    return avg_score, scores


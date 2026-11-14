import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score


def binary_metrics(target, probability, threshold=0.5):
    target, probability = np.asarray(target), np.asarray(probability)
    if set(np.unique(target)) != {0, 1}:
        raise ValueError('AUROC requires both classes')
    tn, fp, fn, tp = confusion_matrix(target, probability >= threshold, labels=[0, 1]).ravel()
    ratio = lambda a, b: float(a/b) if b else 0.0
    return dict(auroc=float(roc_auc_score(target, probability)),
                sensitivity=ratio(tp, tp+fn), specificity=ratio(tn, tn+fp),
                precision=ratio(tp, tp+fp), f1=ratio(2*tp, 2*tp+fp+fn),
                confusion_matrix=[[int(tn), int(fp)], [int(fn), int(tp)]],
                threshold=threshold, zero_division=0)

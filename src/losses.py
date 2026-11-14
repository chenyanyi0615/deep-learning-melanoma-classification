import torch
from torch import nn
from torch.nn import functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super().__init__()
        if gamma < 0 or not 0 <= alpha <= 1:
            raise ValueError('gamma >= 0 and alpha in [0, 1] are required')
        self.gamma, self.alpha = gamma, alpha

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
        pt = torch.exp(-bce)
        alpha_t = self.alpha * target + (1-self.alpha) * (1-target)
        return (alpha_t * (1-pt).pow(self.gamma) * bce).mean()


def make_loss(name, targets, device, gamma=2.0, alpha=0.25):
    positives = int(targets.sum())
    negatives = len(targets) - positives
    if min(positives, negatives) <= 0:
        raise ValueError('Training data must contain both classes')
    if name == 'bce':
        return nn.BCEWithLogitsLoss()
    if name == 'weighted_bce':
        return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negatives/positives, device=device))
    if name == 'focal':
        return BinaryFocalLoss(gamma, alpha)
    raise ValueError(name)

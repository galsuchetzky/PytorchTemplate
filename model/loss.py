import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


def nll_loss():
    return F.nll_loss

import torch.nn.functional as F
from torch.nn import CrossEntropyLoss


def nll_loss():
    return F.nll_loss

# class MaskCrossEntropyLoss(CrossEntropyLoss):
#     def __init__(self):
#         """
#         :param
#         """
#         super().__init__()
#
#     def forward(self, input: Tensor, target: Tensor, mask: Tensor) -> Tensor:
#         nTotal = mask.sum()
#         crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
#         loss = crossEntropy.masked_select(mask).mean()
#         loss = loss.to(device)
#         return loss, nTotal.item()
#         return F.cross_entropy(input, target, weight=self.weight,
#                                ignore_index=self.ignore_index, reduction=self.reduction)
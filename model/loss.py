import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)


def mean_squared_error(output, target):
    return F.mse_loss(output, target)


def binary_cross_entropy_with_logits(output, target):
    return F.binary_cross_entropy_with_logits(output, target)

import torch
import torch.nn.functional as F

def reduce(loss, reduction="mean"):
    if reduction and reduction != "none":
        if reduction == "mean":
            loss = torch.mean(loss)
        elif reduction == "sum":
            loss = torch.sum(loss)
    return loss

def mse(pred, target):
    return ((pred - target) ** 2).mean()
# THIS IS NOT STANDARD MSLE
def mile(pred, y, reduction="mean"):
    error = torch.abs(pred - y)
    loss = mile_(error)
    loss = reduce(loss, reduction=reduction)
    return loss

def mile_(error):
    return torch.log(1+error) * (1+error) - error

def mire(pred, y, scale=0.5, reduction="mean"):
    error = torch.abs(pred - y)
    loss = mire_(error, scale=scale)
    loss = reduce(loss, reduction=reduction)
    return loss

def mire_(error, scale=0.5):
    return scale * torch.pow(error, 1.5)

LOSSES = {
    "bce": F.binary_cross_entropy,
    "mse": F.mse_loss,
    "mae": F.l1_loss,
    "mile": mile,
    "mire": mire,
}

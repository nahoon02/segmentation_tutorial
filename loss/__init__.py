import torch.nn as nn
from loss.dice_loss import DiceLoss
from loss.weighted_bce_loss import WeightedBCELoss

LOSS_CLASSES = {
    'BCE_Loss': nn.BCELoss,
    'DICE_Loss': DiceLoss,
    'WBCE_Loss': WeightedBCELoss,
}
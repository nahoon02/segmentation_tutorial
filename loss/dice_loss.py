import torch
import torch.nn as nn
from torch import Tensor

class DiceLoss(nn.Module):
    """ Batch Dice Loss

    dice loss computed by slice then merged
    """

    def __init__(self, squared_pred: bool = False, smooth: float = 1e-5) -> None:
        """
        Args:
            squared_pred: use squared versions of targets and predictions in the denominator or not.
            smooth: a small constant to avoid zero or nan.
        """
        super(DiceLoss, self).__init__()

        self.smooth = smooth
        self.squared_pred = squared_pred

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: model sigmoid output. shape = (N,1,H,W) or (N,1,D,H,W)
            target: ground truth. shape = (N,1,H,W) or (N,1,D,H,W)
        """

        if pred.dim() != target.dim():
            ValueError(f'[DiceLoss]model output dimension should be same as target dimension')

        """
            squeeze a channel, (N,1,H,W) or (N,1,D,H,W) -> (N,H,W) or (N,D,H,W)
        """
        pred = pred.squeeze(1)
        target = target.squeeze(1)

        if pred.dim() == 3:
            reduce_axis = [1, 2]
        elif pred.dim() == 4:
            reduce_axis = [1, 2, 3]
        else:
            ValueError(f'[DiceLoss] current dimension not supported')

        intersection = torch.sum(target * pred, dim=reduce_axis)

        if self.squared_pred:
            target = torch.pow(target, 2)
            pred = torch.pow(pred, 2)

        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(pred, dim=reduce_axis)

        denominator = ground_o + pred_o

        batch_dice_loss = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth)

        dice_loss = torch.mean(batch_dice_loss)

        return dice_loss
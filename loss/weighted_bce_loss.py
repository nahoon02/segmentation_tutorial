import torch
import torch.nn as nn
from torch import Tensor


class WeightedBCELoss(nn.Module):
    """ when computing bce loss, foreground pixel and background pixels bce have different weights

    loss = f_weight * (foreground pixel loss) + b_weight * (background pixel loss)

    """
    def __init__(self, b_weight: float = 0.1, f_weight: float = 100.) -> None:
        """
        Args:
            b_weight: foreground bce weight
            f_weight: background bce weight
        """
        super(WeightedBCELoss, self).__init__()

        self.b_weight = -b_weight
        self.f_weight = -f_weight
        self.eps = 3.720076e-44

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        """
        Args:
            pred: model output. shape = (N,1,H,W) or (N,1,H,W,D)
            target: ground truth. shape = (N,1,H,W) or (N,1,H,W,D)
        """
        loss_1 = self.f_weight * (target * torch.log(pred + self.eps))
        loss_0 = self.b_weight * ((1 - target) * torch.log(1 - pred + self.eps))

        loss = torch.mean(loss_1 + loss_0)

        return loss


class ForegroundWeightedBCELoss(nn.Module):
    """ same concept of weighted bce loss in torch framework

    foreground weight (f_weight) is automatically determined by foreground and background pixel ratio

    """

    def __init__(self):
        super(ForegroundWeightedBCELoss, self).__init__()

        self.eps = 3.720076e-44

    def forward(self, pred: Tensor, target: Tensor) -> None:
        """
        Args:
            pred: model output. shape = (N,1,H,W) or (N,1,H,W,D)
            target: ground truth. shape = (N,1,H,W) or (N,1,H,W,D)
        """
        """
            compute a foreground weight
        """
        n_foreground_pixels = torch.sum(target)
        n_background_pixels = torch.sum(1.0 - target)

        if n_foreground_pixels == 0.0:
            f_weight = 0.0
        else:
            f_weight = -(n_background_pixels / n_foreground_pixels)

        loss_1 = f_weight * (target * torch.log(pred + self.eps))
        loss_0 = -(1 - target) * torch.log(1 - pred + self.eps)

        loss = torch.mean(loss_1 + loss_0)

        return loss



if __name__ == '__main__':

    input = torch.tensor([
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    ])

    target = torch.tensor([
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    ])

    print(f'target shape = {target.shape}')

    bce = nn.BCELoss()
    b_loss = bce(input, target)
    print(f'bce loss = {b_loss}')

    wbce = WeightedBCELoss()
    w_loss = wbce(input, target)
    print(f'weighted bce loss = {w_loss}')

    wbce2 = ForegroundWeightedBCELoss()
    w2_loss = wbce2(input, target)
    print(f'foreground weighted bce loss = {w2_loss}')




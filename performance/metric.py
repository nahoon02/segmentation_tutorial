import numpy as np
import torch
from typing import Any, Optional
from torch import Tensor

class Metric():
    def __init__(self, callback_fn: Optional[Any] = None, epsilon: float = 1e-7) -> None:
        super(Metric, self).__init__()

        self.callback_fn = callback_fn
        self.epsilon = epsilon

    def __call__(self, pred: Tensor, target: Tensor, **kwargs: Any) -> None:
        """ change Tensor to numpy array and call child function

        Args:
            pred: model sigmoid output. shape = (n,1,h,w) or (n,1,d,h,w)
            target: ground truth
        """
        # post-processing for model output
        if self.callback_fn:
            pred = self.callback_fn(pred, **kwargs)

        # change tensor to numpy array
        if type(target) == torch.Tensor:
            target = target.detach().cpu().numpy()

            if pred is None:
                pred = np.zeros_like(target)
            elif type(pred) == torch.Tensor:
                pred = pred.detach().cpu().numpy()
            else:
                raise RuntimeError(f'target is a tensor but pred is not a tensor')

        self.apply(pred, target, **kwargs)

    def apply(self, pred, target, **kwargs):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError

    def reset(self):
        raise NotImplementedError
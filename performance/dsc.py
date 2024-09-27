from performance.metric import Metric
import numpy as np
from typing import Optional, Any, List


class DSC(Metric):
    """ compute Dice score coefficient
    """
    def __init__(self, threshold: float = 0.5, mode: str = 'batch', callback_fn: Optional[Any] = None,
                 epsilon: float = 1e-7) -> None:
        """
        Args:
            threshold: before DSC, model output should be binarized using threshold.
            mode: DSC operation mode. ['batch', 'slice', 'monai']
            callback_fn: processing callback_fn followed by DSC operation.
            epsilon: prevent nan or zero
        """
        super(DSC, self).__init__(callback_fn, epsilon)

        self.mode = mode
        self.threshold = threshold
        self.dsc_list = []

    def apply(self, pred: np.ndarray, target: np.ndarray, **kwargs: Any) -> None:
        """
        Args:
            pred: model output, shape = (N,1,H,W) or (N,1,D,H,W)
            target: GT, shape = (N,1,H,W) or (N,1,D,H,W)
        """
        assert type(pred) == np.ndarray
        assert type(target) == np.ndarray

        if self.mode == 'batch':
            dsc = self.compute_dsc(pred, target)

        self.dsc_list.append(dsc)

    def get_mode(self):
        return self.mode

    def get(self):
        return self.dsc_list[-1]

    def result(self):

        dsc_np = self.dsc_list

        # remove nan elements and compute average
        dsc = np.nanmean(dsc_np)
        if dsc == np.nan:
            dsc = 0.0

        ret = {'dsc': float('%.3f' % dsc)}

        # clear dsc list
        self.reset()

        return ret

    def reset(self):
        self.dsc_list.clear()

    def compute_dsc(self, pred, target):
        """
        Args:
            pred: model output, shape = (N,1,H,W) or (N,1,D,H,W)
            target: GT, shape = (N,1,H,W) or (N,1,D,H,W)
        """
        assert pred.shape == target.shape
        assert pred.shape[1] == 1
        assert target.shape[1] == 1

        # remove channel axis (N,1,H,W) or (N,1,D,H,W) --> (N,H,W) or (N,D,H,W)
        pred = np.squeeze(pred, axis=1)
        target = np.squeeze(target, axis=1)

        # if shape is (N, H, W), add D = 1 dimension. (N, H, W) --> (N, D=1, H, W)
        # So (N,1,H,W) or (N,1,D,H,W) has a same shape (N,D,H,W)
        if pred.ndim == 3:
            pred = np.expand_dims(pred, axis=1)
            target = np.expand_dims(target, axis=1)

        # binarize
        pred = (pred > self.threshold).astype(np.float32)
        target = target.astype(np.float32)

        # flatten
        pred = pred.flatten()
        target = target.flatten()

        intersection = np.sum(pred * target)

        y_o = np.sum(target) # GT
        y_pred_o = np.sum(pred) # prediction
        denominator = y_o + y_pred_o

        dsc = (2.0 * intersection + self.epsilon) / (denominator + self.epsilon)

        if y_o < 1.0: # background slice is excluded in dsc measure
            dsc = np.nan

        return dsc
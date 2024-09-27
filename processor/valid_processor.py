import torch
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from performance.metric import Metric
from torch.utils.data import DataLoader
import torch.nn as nn
from typing import List


def valid_processor(epoch: int, model: nn.Module, data_loader: DataLoader, measure: Metric, use_cuda: bool) -> List:
    """ executed every epoch
    Args:
        epoch: current epoch index
        model: model instance
        data_loader: dataloader instance
        measure_list: measure class instance list
        use_cuda: if True, use GPU
    """

    model.eval()

    num_iter = len(data_loader)

    # progress bar
    t = tqdm(total=num_iter)

    with torch.no_grad():

        for i, (images, masks, _, _, _) in enumerate(data_loader):

            # tensor moves to cuda device
            if use_cuda:
                images = images.cuda()
                masks = masks.cuda()

            # run model
            masks_pred = model(images)
            if isinstance(masks_pred, tuple):
                masks_pred, _ = masks_pred
            else:
                masks_pred = masks_pred

            n, c, h, w = masks.size()
            assert n == 1 and c == 1

            # -----------------------------------------------------------------------
            # masks_pred (model output) shape = (1, 1, resized_h, resized_w)
            # maks (GT) shape = (1, 1, resized_h, resize_w)
            # for validation, two shape should be identical
            # ---------------------------------------------------------------------
            # performance
            measure(masks_pred, masks)

            # display progress bar
            print_str = f'[valid][epoch:{epoch}]- [{i + 1}/{num_iter}]'
            t.set_description(print_str)
            t.update(1)

        # close tqdm
        t.close()

    ret = measure.result()

    return ret['dsc']

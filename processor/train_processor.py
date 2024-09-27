import torch
import os
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.optim import Optimizer


def train_processor(epoch: int, model: nn.Module, data_loader: DataLoader, optimizer: Optimizer,
                    criterion: nn.Module, use_cuda: bool) -> float:
    """ executed every epoch

    Args:
        epoch: current epoch index
        model: model instance
        data_loader: DataLoader instance
        optimizer: Optimizer instance
        criterion: loss class instance
        use_cuda: if True : use GPU
    """

    model.train()

    num_iter = len(data_loader)
    epoch_loss = 0.0

    # progress bar
    t = tqdm(total=num_iter)

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

        loss = criterion(masks_pred, masks)

        epoch_loss += loss.item()

        # update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # display progress bar
        print_str = f'[train][epoch:{epoch}]- [{i+1}/{num_iter}] loss=[{loss.item():.3f}]'
        t.set_description(print_str)
        t.update(1)

    average_loss = epoch_loss/num_iter

    # close tqdm
    t.close()

    return average_loss

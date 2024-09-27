import os
import torch
from datetime import datetime
import torch.nn as nn
from typing import Optional, List, Tuple


def save_checkpoint(model_name: str, epoch: int, checkpoints_dir: str, model: nn.Module):
    """ save model weights

    Args:
        model_name: model instance name
        epoch: current epoch index
        checkpoints_dir: save directory
        model: model instance
    """
    # save model parameters
    now = datetime.now()
    now = now.strftime('%m%d%H%M')
    model_filename = model_name + '_' + 'epoch_' + f'{epoch:03d}' + '_' + now + '.pth'
    model_savepath = os.path.join(checkpoints_dir, model_filename)

    state = {'epoch': epoch, 'state_dict': model.state_dict()}
    torch.save(state, model_savepath)

    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print(f'checkpoint saved --> {model_savepath}')
    print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++')

    return model_filename


def load_checkpoint(model: nn.Module, checkpoint_filename: str, device: str='cuda:0') -> Tuple[nn.Module, int]:

    if os.path.isfile(checkpoint_filename):
        checkpoint = torch.load(checkpoint_filename, map_location=device)
        epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        raise RuntimeError(f'checkpoint file = [{checkpoint_filename}] does not exist')

    return model, epoch



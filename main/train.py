import os
import sys
import time
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from config.default_config import get_cfg_defaults, tuple_to_dict
from model import MODEL_CLASSES
from loss import LOSS_CLASSES
import numpy as np
import matplotlib.pyplot as plt
from util.logger import get_logger
from util.checkpoint import save_checkpoint
from optimizer import OPTIMIZER_CLASSES
from util.time_conversion import convert_second_to_time
from util.graph import display_graph_v2
import math
from performance.dsc import DSC
from dataloader import DATASET_CLASSES
from processor import PROCESSOR_CLASSES
import warnings
warnings.filterwarnings('ignore')


def main(yaml_path: str) -> None:
    """
    ARGS:
        yaml_path: yaml file full path

    example: train.py ../config/xxx.yaml
    """

    print('current directory = ', os.getcwd())

    """ configuration """
    # load default config
    cfg = get_cfg_defaults()

    cfg_filepath = os.path.expanduser(yaml_path)

    print(f'config file -> {cfg_filepath}')
    # load custom config
    if os.path.isfile(cfg_filepath):
        cfg.merge_from_file(cfg_filepath)
        cfg.CUSTOM_CONFIG.FILENAME = cfg_filepath
    else:
        raise RuntimeError(f'config file [{cfg_filepath}] does not exist')

    # record start time
    now = datetime.now()
    now = now.strftime('%Y-%m-%d %H:%M')
    cfg.REPORT.START_TIME = now
    # cfg.freeze()
    """"""

    """ check GPUs """
    if torch.cuda.is_available():
        torch.cuda.set_device(cfg.CUDA.DEVICE_ID)
        device = 'cuda:' + str(cfg.CUDA.DEVICE_ID)
        use_cuda = True
        print('Available GPU devices ==> ', torch.cuda.device_count())
        print('Current cuda device ==> ', torch.cuda.current_device())
    else:
        device = 'cpu'
        use_cuda = False



    """ check directories """
    ct_data_dir = os.path.expanduser(cfg.DATASET.CT_DIR_ROOT)
    mask_data_dir = os.path.expanduser(cfg.DATASET.MASK_DIR_ROOT)
    # get yaml filename
    yaml_filename, _ = os.path.splitext(os.path.basename(cfg_filepath))

    # checkpoint directory modified
    cfg.CHECKPOINTS.SAVE_DIR = os.path.join(os.path.expanduser(cfg.CHECKPOINTS.SAVE_DIR), yaml_filename)
    save_checkpoints_dir = cfg.CHECKPOINTS.SAVE_DIR
    # report directory modified
    cfg.REPORT.TRAIN_DIR = os.path.join(os.path.expanduser(cfg.REPORT.TRAIN_DIR), yaml_filename)
    report_dir = cfg.REPORT.TRAIN_DIR

    if not os.path.isdir(ct_data_dir):
        print(f'ct data dir=[{ct_data_dir}] does not exist.')
        exit(1)
    if not os.path.isdir(mask_data_dir):
        print(f'mask data dir=[{ct_data_dir}] does not exist.')
        exit(1)
    if not os.path.isdir(save_checkpoints_dir):
        print(f'checkpoint directory=[{save_checkpoints_dir}]  does not exist.')
        os.makedirs(save_checkpoints_dir, exist_ok=True)
        print(f'checkpoint directory=[{save_checkpoints_dir}] is created !!!!')
    if not os.path.isdir(report_dir):
        print(f'report directory=[{report_dir}] does not exist.')
        os.makedirs(report_dir, exist_ok=True)
        print(f'report directory=[{report_dir}] is created !!!!')

    csv_file = os.path.expanduser(cfg.DATASET.CSV_FILE)
    if not os.path.isfile(csv_file):
        print(f'csv file=[{csv_file}] does not exist.')
        exit(1)

    """ define loaders """
    dict_params = tuple_to_dict(cfg.DATASET.TRAIN_PARAMS)
    train_dataset = DATASET_CLASSES[cfg.DATASET.TRAIN_NAME](**dict_params)
    print(f'train dataset param = {dict_params}')

    batch_size = cfg.HYPER_PARAMS.BATCH_SIZE
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=cfg.DATALOADER.NUM_WORKERS, drop_last=True)

    dict_params = tuple_to_dict(cfg.DATASET.VALID_PARAMS)
    valid_dataset = DATASET_CLASSES[cfg.DATASET.VALID_NAME](**dict_params)
    print(f'valid dataset param = {dict_params}')
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS)

    """ create a model """
    dict_params = tuple_to_dict(cfg.MODEL.PARAMS)
    model = MODEL_CLASSES[cfg.MODEL.NAME](**dict_params)
    model_name = cfg.MODEL.NAME
    print(f'model param = {dict_params}')

    """ device setting """
    if use_cuda:
        model.cuda()

    """ create a criterion """
    dict_params = tuple_to_dict(cfg.MODEL.LOSS_PARAMS)
    criterion = LOSS_CLASSES[cfg.MODEL.LOSS](**dict_params)
    loss_param = str(dict_params)

    """ create an optimizer """
    dict_params = tuple_to_dict(cfg.MODEL.OPTIMIZER_PARAMS)
    optimizer = OPTIMIZER_CLASSES[cfg.MODEL.OPTIMIZER](model.parameters(), lr=cfg.HYPER_PARAMS.LEARNING_RATE,
                                                       **dict_params)

    """ create a performance measure """
    measure = DSC()

    """ logger """
    # make a log file
    logger = get_logger(report_dir, 'train_result_' + cfg.MODEL.NAME)
    logger.info('----------- [config settings] ----------------------')
    logger.info(cfg)
    logger.info('number of train data')
    logger.info(train_dataset.get_info())
    logger.info('number of valid data')
    logger.info(valid_dataset.get_info())
    logger.info('----------------------------------------------------')

    """ select train/valid processor"""
    processor_train = PROCESSOR_CLASSES[cfg.MODEL.TRAIN_PROCESSOR]
    processor_valid = PROCESSOR_CLASSES[cfg.MODEL.VALID_PROCESSOR]

    plt.ion()
    plt.figure(figsize=(11, 10))  # figure size = (width, height)
    train_loss_list = []
    valid_dsc_list = []

    start_time = time.time()
    start_epoch = 1
    for epoch in range(1, cfg.HYPER_PARAMS.NUM_EPOCHS + 1):  # (start_epoch ~ NUM_EPOCH)
        since = time.time()

        """ train """
        train_loss = processor_train(epoch, model, train_loader, optimizer, criterion, use_cuda)
        if math.isnan(train_loss):
            log = f'epoch = [{epoch}] nan happens !!!!!'
            logger.error(log)
            sys.exit()

        """ valid """
        valid_dsc = processor_valid(epoch, model, valid_loader, measure, use_cuda)

        """ save train/valid history """
        train_loss_list.append(train_loss)
        valid_dsc_list.append(valid_dsc)

        seconds = time.time() - since
        time_elapsed = convert_second_to_time(seconds)

        """ save a model checkpoint  """
        model_filename = save_checkpoint(model_name, epoch, save_checkpoints_dir, model)

        """ display a graph"""
        title = model_name + '  ' + cfg.MODEL.LOSS + loss_param

        display_graph_v2(plt, title, start_epoch, train_loss_list, cfg.HYPER_PARAMS.NUM_EPOCHS,
                         save_graph=True, model_filename=model_filename, save_dir=report_dir,
                         valid_dsc=valid_dsc_list)

        """ find out best epoch, best value """
        best_indices, best_values = get_best(valid_dsc_list, start_epoch, 1)

        """ logging """
        log = f'epoch,{epoch:3d},train loss,{train_loss:.3f},' \
              f'valid:(dsc),{valid_dsc:.3f},' \
              f'elapsed,{time_elapsed},{model_filename},' \
              f'best epoch,{best_indices[0]},best DSC,{best_values[0]:.3f}'
        logger.info(log)

        """ time remained"""
        remained_epoch = cfg.HYPER_PARAMS.NUM_EPOCHS - epoch
        remained_time = convert_second_to_time(remained_epoch * seconds)
        elapsed_time = convert_second_to_time(epoch * seconds)
        print(f'elapsed time = {elapsed_time}, remained time = {remained_time}')

    total_seconds = time.time() - start_time
    duration = convert_second_to_time(total_seconds)

    """ find out best epoch, best value """
    best_indices, best_values = get_best(valid_dsc_list, start_epoch, 3)
    logger.info('------------ [best DSC in valid_dataset] ------------')
    for idx, value in zip(best_indices, best_values):
        logger.info('best epoch[{}]=[{:.3f}]'.format(idx, value))
    logger.info('---------------------------------------')

    ''' record elapsed time '''
    logger.info(f'elapsed train time: {duration}')


def get_best(value_list, start_idx=1, top_n=3):
    value_array = np.array(value_list)
    best_indices = np.argsort(value_array)[::-1][:top_n]  # descending order
    best_values = value_array[best_indices]
    # shift idx
    best_indices = best_indices + start_idx

    return best_indices.tolist(), best_values.tolist()


if __name__ == '__main__':

    if len(sys.argv) == 2 and sys.argv[1]:
        main(sys.argv[1])
    else:
        main('./config/unet_0927.yaml')

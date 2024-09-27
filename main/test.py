import os
import sys
sys.path.append('..')
import time
import torch
from torch.utils.data import DataLoader
from datetime import datetime
from config.default_config import get_cfg_defaults, tuple_to_dict
from model import MODEL_CLASSES
from util.logger import get_logger
from util.checkpoint import load_checkpoint
from util.time_conversion import convert_second_to_time
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

    # checkpoint path
    checkpoint_path = os.path.expanduser(cfg.TEST.CHECKPOINT_FILE)

    # report directory modified
    cfg.REPORT.TRAIN_DIR = os.path.join(os.path.expanduser(cfg.REPORT.TRAIN_DIR), yaml_filename)
    report_dir = cfg.REPORT.TRAIN_DIR

    if not os.path.isdir(ct_data_dir):
        print(f'ct data dir=[{ct_data_dir}] does not exist.')
        exit(1)
    if not os.path.isdir(mask_data_dir):
        print(f'mask data dir=[{ct_data_dir}] does not exist.')
        exit(1)
    if not os.path.isfile(checkpoint_path):
        print(f'checkpoint file=[{checkpoint_path}]  does not exist.')
        exit(1)
    if not os.path.isdir(report_dir):
        print(f'report directory=[{report_dir}] does not exist.')
        os.makedirs(report_dir, exist_ok=True)
        print(f'report directory=[{report_dir}] is created !!!!')

    csv_file = os.path.expanduser(cfg.DATASET.CSV_FILE)
    if not os.path.isfile(csv_file):
        print(f'csv file=[{csv_file}] does not exist.')
        exit(1)

    output_dir = os.path.join(report_dir, 'output')
    if not os.path.isdir(output_dir):
        print(f'output directory=[{output_dir}] does not exist.')
        os.makedirs(output_dir, exist_ok=True)
        print(f'output directory=[{output_dir}] is created !!!!')

    """ define loaders """
    dict_params = tuple_to_dict(cfg.DATASET.TEST_PARAMS)
    test_dataset = DATASET_CLASSES[cfg.DATASET.TEST_NAME](**dict_params)
    print(f'test dataset param = {dict_params}')
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=cfg.DATALOADER.NUM_WORKERS)

    """ create a model """
    dict_params = tuple_to_dict(cfg.MODEL.PARAMS)
    model = MODEL_CLASSES[cfg.MODEL.NAME](**dict_params)
    model_name = cfg.MODEL.NAME
    print(f'model param = {dict_params}')

    """ load a checkpoint file """
    model, _ = load_checkpoint(model, checkpoint_path, device)

    """ device setting """
    if use_cuda:
        model.cuda()

    """ create a performance measure """
    measure = DSC()

    """ logger """
    # make a log file
    logger = get_logger(report_dir, 'test_result_' + cfg.MODEL.NAME)
    logger.info('----------- [config settings] ----------------------')
    logger.info(cfg)
    logger.info('number of test data')
    logger.info(test_dataset.get_info())
    logger.info('----------------------------------------------------')

    """ select train/valid processor"""
    processor = PROCESSOR_CLASSES[cfg.MODEL.TEST_PROCESSOR]

    since = time.time()

    """ test """
    dsc = processor(output_dir, model, test_loader, measure, use_cuda)

    seconds = time.time() - since
    time_elapsed = convert_second_to_time(seconds)

    """ logging """
    log = f'checkpoint file,{checkpoint_path},test:(dsc),{dsc:.3f},elapsed,{time_elapsed}'
    logger.info(log)


if __name__ == '__main__':

    if len(sys.argv) == 2 and sys.argv[1]:
        main(sys.argv[1])
    else:
        main('../config/unet_0927.yaml')

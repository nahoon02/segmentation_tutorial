from yacs.config import CfgNode as CN

_C = CN()
# ---------------------------------------------------------------------------- #
# Hyper parameters
# ---------------------------------------------------------------------------- #
_C.HYPER_PARAMS = CN()
_C.HYPER_PARAMS.LEARNING_RATE = 0.01
_C.HYPER_PARAMS.NUM_EPOCHS = 50
_C.HYPER_PARAMS.BATCH_SIZE = 4

# ---------------------------------------------------------------------------- #
# CUDA Settings
# ---------------------------------------------------------------------------- #
_C.CUDA = CN()
_C.CUDA.DEVICE_ID = 0

# ---------------------------------------------------------------------------- #
# dataset
# ---------------------------------------------------------------------------- #
_C.DATASET = CN()
_C.DATASET.CT_DIR_ROOT = ''
_C.DATASET.MASK_DIR_ROOT = ''

# csv file path
_C.DATASET.CSV_FILE = 'dataset.csv'

_C.DATASET.TRAIN_NAME = 'TrainDataset'
_C.DATASET.TRAIN_PARAMS = ()
_C.DATASET.VALID_NAME = 'TrainDataset'
_C.DATASET.VALID_PARAMS = ()
_C.DATASET.TEST_NAME = 'TrainDataset'
_C.DATASET.TEST_PARAMS = ()

# ---------------------------------------------------------------------------- #
# data loader
# ---------------------------------------------------------------------------- #
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4

# ---------------------------------------------------------------------------- #
# Model parameters
# ---------------------------------------------------------------------------- #
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = 'unet'
_C.MODEL.PARAMS = ()

_C.MODEL.OPTIMIZER = 'SGD'
_C.MODEL.OPTIMIZER_PARAMS = ('momentum', 0.9)

_C.MODEL.LOSS = 'BCE_Loss'
_C.MODEL.LOSS_PARAMS = ()

_C.MODEL.TRAIN_PROCESSOR = 'train_processor'
_C.MODEL.VALID_PROCESSOR = 'valid_processor'
_C.MODEL.TEST_PROCESSOR = 'test_processor'

# ---------------------------------------------------------------------------- #
# Performance
# ---------------------------------------------------------------------------- #
_C.PERORMANCE = CN()
_C.PERORMANCE.MEASURE = ('Accuracy_slice', 'DSC')
_C.PERORMANCE.PARAMS = (('threshold',0.5,'callback_fn', None), ('threshold',0.5,'mode','batch'))

# ---------------------------------------------------------------------------- #
# Checkpoints
# ---------------------------------------------------------------------------- #
_C.CHECKPOINTS = CN()
_C.CHECKPOINTS.SAVE_DIR = 'none'

# ---------------------------------------------------------------------------- #
# test
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
_C.TEST.CHECKPOINT_FILE = 'none'

# ---------------------------------------------------------------------------- #
# report
# ---------------------------------------------------------------------------- #
_C.REPORT = CN()
_C.REPORT.TRAIN_DIR = 'none'
_C.REPORT.TEST_DIR = 'none'
_C.REPORT.OWNER = 'ai_health'
_C.REPORT.START_TIME = 'none'

# ---------------------------------------------------------------------------- #
# custom config
# ---------------------------------------------------------------------------- #
_C.CUSTOM_CONFIG = CN()
_C.CUSTOM_CONFIG.FILENAME = 'none'

def get_cfg_defaults():
    return _C.clone()

# config does not support dict type. Use tuple instead of dict !!!
def tuple_to_dict(tuple_param):

    if len(tuple_param) == 0:
        return {}

    if len(tuple_param) % 2 != 0:
        raise RuntimeError(f'tuple param {tuple_param} should have multiples of 2')

    dict_size = len(tuple_param) // 2
    dict_param = {}
    for i in range(dict_size):
        dict_param[tuple_param[i * 2]] = tuple_param[i * 2 + 1]

    return dict_param
import logging
import logging.handlers
from datetime import datetime
import os

def get_logger(dir: str, filename: str):
    """ when a log is saved, log info is printed on console
    Args:
        dir: saved directory
        filename: log file name without extension
    """
    # define log file name
    now = datetime.now()
    now = now.strftime('%Y%m%d%H%M')

    # make log file name
    if filename.find('.txt') != -1:
        filename = filename.replace('.txt', '')
    log_file_name = filename + '_' + now + '.txt'
    log_file_full_path = os.path.join(dir, log_file_name)

    # create logger
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('%(message)s')

    # file handler
    fileHandler = logging.FileHandler(log_file_full_path)
    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)

    # stdout handler
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(formatter)
    logger.addHandler(consoleHandler)

    # logger level
    logger.setLevel(level=logging.DEBUG)

    return logger

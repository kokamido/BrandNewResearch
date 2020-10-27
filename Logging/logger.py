from os import path, mkdir
import logging
from logging.handlers import TimedRotatingFileHandler


def setup_custom_logger(name):
    #logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[TimedRotatingFileHandler(name, when="midnight", interval=1)])
    logger = logging.getLogger('sas')
    logger.setLevel(logging.INFO)
    return logger


if not path.exists('Logs'):
    mkdir('Logs')
log = setup_custom_logger('Logs/Log')

from os import path, mkdir
import logging
from logging.handlers import TimedRotatingFileHandler


def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = TimedRotatingFileHandler(name, when="midnight", interval=1)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


if not path.exists('Logs'):
    mkdir('Logs')
log = setup_custom_logger('Logs/Log')

import logging
from logging.handlers import TimedRotatingFileHandler
import time

def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(module)s - %(message)s')

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    handler = TimedRotatingFileHandler(name, when="S", interval=1)
    logger.addHandler(handler)
    return logger

logger = setup_custom_logger('logs/memes')


for i in range(6):
    logger.info(f'{i}')
    time.sleep(.5)
    logger.error(f'{i}')
    time.sleep(.5)
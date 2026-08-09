import logging


def get_logger(fmt, name=None, level=logging.DEBUG):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    logger_handler = logging.StreamHandler()
    logger_handler.setLevel(level)
    logger_handler.setFormatter(fmt())

    logger.addHandler(logger_handler)

    return logger

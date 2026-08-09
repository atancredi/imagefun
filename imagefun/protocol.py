from logging import Logger
from PIL import Image


class ImagefunProtocol:
    image: Image.Image
    logger: Logger

    def log(self, msg: str, level: str = "info", extra=None):
        if self.logger:
            self.logger.log(level, msg, extra)

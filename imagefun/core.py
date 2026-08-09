import logging
from io import BytesIO
from typing import Optional, Callable, Any

from PIL import Image

from .logger import get_logger, PrettyFormatter
from .mixins import (
    ImagefunAnalysisMixin,
    ImagefunColorMixin,
    ImagefunDitheringMixin,
    ImagefunResizeMixin,
)


class ImagefunContext:
    def __init__(
        self,
        image: Image.Image,
        path: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self.image = image
        self.path = path
        self.logger = logger
        self.history: list = []

    # TODO REMOVE THIS CMON
    def run_if_condition(
        self, condition: bool, function: Callable[["Imagefun"], Any]
    ) -> "Imagefun":
        if condition:
            function(self)
        return self


class Imagefun(
    ImagefunContext,
    ImagefunAnalysisMixin,
    ImagefunColorMixin,
    ImagefunDitheringMixin,
    ImagefunResizeMixin,
):
    def save(self, output_path: str, optimize: bool = False) -> "Imagefun":
        self.image.save(output_path, optimize=optimize)
        self.log(f"Saved image to {output_path}")
        return self


class ImagefunLoader:
    @staticmethod
    def from_file(path, logger: logging.Logger = None):
        if logger == None:
            logger = get_logger(PrettyFormatter, level=logging.INFO)
        i = Imagefun(logger)
        i.image = Image.open(path)
        i.path = path
        i.log(
            "[IMG_LOADED] Loaded image from file.", level="info", extra={"path": path}
        )
        return i

    @staticmethod
    def from_image(image: Image.Image, logger: logging.Logger = None):
        if logger == None:
            logger = get_logger(PrettyFormatter, level=logging.INFO)
        i = Imagefun(logger)
        i.image = image
        i.log("[IMG_LOADED] Loaded image from PIL Image.", level="info")
        return i

    @staticmethod
    def from_bytes(b: BytesIO, logger: logging.Logger = None):
        if logger == None:
            logger = get_logger(PrettyFormatter, level=logging.INFO)
        i = Imagefun(logger)
        i.image = Image.open(b)
        i.log("[IMG_LOADED] Loaded image from bytes.", level="info")
        return i

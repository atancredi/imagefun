import os

from PIL import Image
from .core import Imagefun


class Resize(Imagefun):

    def __init__(self):
        super().__init__()
    
    def _resize(self, new_size):
        self.image = self.image.resize(new_size, Image.Resampling.LANCZOS)
        self.load_image()

        if self.logger:
            self.logger.info(f"Resized image to {new_size}")

    def resize_linked(self, target: int):
        original_width, original_height = self.image.size
        ratio = original_height / original_width

        new_size = (target, int(target * ratio))
        self._resize(new_size)

        return self

    def resize_by_factor(self, factor: float):
        
        original_width, original_height = self.image.size
        new_size = (int(original_width * factor), int(original_height * factor))
        self._resize(new_size)

        return self

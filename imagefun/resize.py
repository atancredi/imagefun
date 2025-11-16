import os

from PIL import Image
from .core import Imagefun


class Resize(Imagefun):

    def __init__(self, properties = None):
        super().__init__(properties)
        self._ignore = False
    
    def ignore(self, ignore: bool):
        self._ignore = ignore

    def resize_image_proportional(self, target_width=None, target_height=None):
        original_width, original_height = self.image.size
        ratio = original_width / original_height

        if target_width is not None:
            new_height = int(original_height * ratio)
            new_size = (target_width, new_height)

        elif target_height is not None:
            new_width = int(original_width * ratio)
            new_size = (new_width, target_height)

        else:
            return self

        self.image = self.image.resize(new_size, Image.Resampling.LANCZOS)
        return self
    
    def resize_image_larger_side(self, target: int):
        original_width, original_height = self.image.size
        ratio = original_height / original_width

        # get larger side and set it to target mantaining the ratio
        if original_height >= original_width:
            new_size = (int(target * ratio), target)
        else:
            new_size = (target, int(target * ratio))

        self.image = self.image.resize(new_size, Image.Resampling.LANCZOS)
        return self

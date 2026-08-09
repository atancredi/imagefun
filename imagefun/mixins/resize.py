from typing import Tuple
from PIL import Image


from ..protocol import ImagefunProtocol

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..core import Imagefun


class ImagefunResizeMixin(ImagefunProtocol):
    def _resize(self, new_size: Tuple[int, int]) -> "Imagefun":
        self.image = self.image.resize(new_size, Image.Resampling.LANCZOS)
        self.log(f"Resized image to {new_size}")
        return self

    def resize_linked(self, target: int) -> "Imagefun":
        original_width, original_height = self.image.size
        ratio = original_height / original_width
        new_size = (target, int(target * ratio))
        return self._resize(new_size)

    def resize_by_factor(self, factor: float) -> "Imagefun":
        original_width, original_height = self.image.size
        new_size = (int(original_width * factor), int(original_height * factor))
        return self._resize(new_size)

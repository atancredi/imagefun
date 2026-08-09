from PIL import ImageOps

from ..protocol import ImagefunProtocol

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..core import Imagefun


class ImagefunColorMixin(ImagefunProtocol):
    def invert(self) -> "Imagefun":
        self.image = ImageOps.invert(self.image.convert("L"))
        self.log("Inverted image colors")
        return self

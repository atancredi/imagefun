from math import sqrt
from PIL import ImageStat

from ..protocol import ImagefunProtocol


class ImagefunAnalysisMixin(ImagefunProtocol):
    @property
    def size(self):
        # (width, height)
        return self.image.size

    @property
    def brightness(self):
        brightness_magic_values = (0.299, 0.587, 0.114)
        stat = ImageStat.Stat(self.image)
        channels = stat.mean
        return sqrt(
            sum([x * (y**2) for x, y in zip(brightness_magic_values, channels)], 0)
        )

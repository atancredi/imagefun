from typing import Optional
from dataclasses import dataclass

from PIL import Image

# TODO convert leading space to tabs?!?!?!?!?!? WTF IST THIS???!?!??!?!?!

@dataclass
class ImageProperties:
    width: int
    height: Optional[int]


class Imagefun:

    def __init__(self, filename: str, properties: ImageProperties):
        self.filename = filename
        self.image = Image.open(filename)

        height = int(self.image.height * properties.width / self.image.width)
        width = properties.width
        self.image = self.image.resize((width, height))

	###### UTILITY METHODS ####################################################################################

    def save(self, output_name: Optional[str] = None):
        if output_name != None and output_name != "":
            self.image.save(output_name)
        else:
            _ = self.filename.split(".")
            f = _[0] + "_" + "edit" + "." + _[1]
            self.image.save(f)

	###### UTILITY FILTERS #####################################################################################

    def check_square(self):
        pass

    def add_watermark(self):
        pass

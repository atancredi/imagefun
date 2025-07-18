from typing import Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
from math import sqrt

from tqdm import tqdm
from PIL import Image, ImageStat


class ColorSpaces(Enum):
	RGB = "RBG"


@dataclass
class ImageProperties:
	width: Optional[int] = None
	height: Optional[int] = None
	color_space: ColorSpaces = ColorSpaces.RGB


brightness_magic_values = (0.299, 0.587, 0.114)
class Imagefun:
	image: Image.Image

	def __init__(self, properties=None):
		self.properties = properties or {}
		self.filters = []
		self.image = None
		self.pixels = None
		self.width = 0
		self.height = 0

	def load_image(self):
		self.pixels = self.image.load()
		self.width, self.height = self.image.size

	def update_with_properties(self):
		if self.properties is not None:
			if self.properties.color_space == ColorSpaces.RGB:
				self.image.convert("RGB")
			if self.properties.width != None:
				width = self.properties.width
				if self.properties.height != None:
					height = self.properties.height
				else:
					height = int(self.image.height * width / self.image.width)

				self.image = self.image.resize((width, height))

	def from_file(self, path):
		self.image = Image.open(path)
		self.update_with_properties()
		self.load_image()
		return self

	def from_image(self, image: Image.Image):
		self.image = image
		self.update_with_properties()
		self.load_image()
		return self

	def __init__(self, properties: Optional[ImageProperties] = None):
		self.properties = properties
		self.filters = []

	# IMAGE FUNCTIONS
	def add_filter(self, func, **kwargs):
		"""Add a filter function with parameters"""

		def wrapped(pixel):
			return func(pixel, **kwargs)

		self.filters.append(wrapped)
		return self

	def process(self):
		for y in tqdm(range(self.height)):
			for x in tqdm(range(self.width), leave=False):
				pixel = self.pixels[x, y]
				for f in self.filters:
					pixel = f(pixel)
				self.pixels[x, y] = pixel
		return self
	
	def run_manipulation(self, f, **kwargs):
		self.image = f(self.image, **kwargs)
		self.load_image()
		return self

	def save(self, output_path: Optional[str] = None):
		self.image.save(output_path)
		return self

	# ANALYSIS FUNCTIONS
	def print_brightness(self):
		stat = ImageStat.Stat(self.image)
		channels = stat.mean
		self._brightness = sqrt(
			sum([x * (y**2) for x, y in zip(brightness_magic_values, channels)], 0)
		)
		print(self._brightness)
		return self

from typing import Optional
from dataclasses import dataclass
from enum import Enum
from math import sqrt
import numpy as np
from typing import Callable, Self
from logging import Logger

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
class Imagefun(object):
	image: Image.Image
	logger: Logger
	path: str

	def __init__(self, properties=None):
		self.properties = properties or {}
		self.filters = []
		self.image = None

		self.pixels = None
		self.width = 0
		self.height = 0

		self.logger = None
	
	def set_logger(self, logger: Logger):
		self.logger = logger
		return self

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

	# XXX can make the constructors better with a classmethod
	def from_file(self, path):
		self.image = Image.open(path)
		self.path = path
		self.update_with_properties()
		self.load_image()
		return self

	def from_image(self, image: Image.Image):
		self.image = image
		self.update_with_properties()
		self.load_image()
		return self
	
	@classmethod
	def from_instance(cls, instance: Self):
		i = cls()
		i.image = instance.image
		i.properties = instance.properties
		i.logger = instance.logger
		i.load_image()
		return i


	# IMAGE FUNCTIONS
	def add_pixel_filter(self, func, **kwargs):
		"""
			Add a pixel_filter function.\n
			func ( (**kwargs) -> (float, float, float) ): function that elaborates the pixel and returns the 3 RGB channels of that pixel.
		"""

		def wrapped(pixel):
			return func(pixel, **kwargs)

		self.filters.append(wrapped)
		return self

	def process_pixels(self):
		"""Process the pixel of the image using the added pixel_filters"""
		for y in tqdm(range(self.height)):
			for x in tqdm(range(self.width), leave=False):
				pixel = self.pixels[x, y]
				for f in self.filters:
					pixel = f(pixel)
				self.pixels[x, y] = pixel
		return self
	
	def run_manipulation(self, func, **kwargs):
		"""
			Run a function that manipulates the whole image\n
			func ( (image: Image, **kwargs) -> Image )
		"""
		self.image = func(self.image, **kwargs)
		self.load_image()
		return self
	
	
	# DECORATORS
	def effect(func):
		def wrapper(self):
			func(self)
		return func
	

	def run_function(self, func, **kwargs):
		"""
			Run a function that exposes the instance of the class
		"""
		func(self, **kwargs)
		return self

	def save(self, output_path: str, optimize=False):
		"""Save the image to 'output_path'"""
		self.image.save(output_path, optimize=optimize)
		return self


	# CONDITIONAL
	def conditional(self, condition, function: Callable[[Self], Self]):
		if condition:
			function(self)
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

	# PROPERTIES
	@property
	def size(self):
		return (self.width, self.height)
	

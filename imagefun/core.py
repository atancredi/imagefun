from enum import Enum
from math import sqrt
from typing import Callable, Self, List, Any
from logging import Logger
import matplotlib.pyplot as plt

from tqdm import tqdm
from PIL import Image, ImageStat


class ColorSpaces(Enum):
	RGB = "RBG"



brightness_magic_values = (0.299, 0.587, 0.114)
class Imagefun(object):
	image: Image.Image
	logger: Logger
	path: str

	def __init__(self):
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

	# XXX can make the constructors better with a classmethod
	@classmethod
	def from_file(cls, path):
		i = cls()
		i.image = Image.open(path)
		i.path = path
		i.load_image()
		return i

	@classmethod
	def from_image(cls, image: Image.Image):
		i = cls()
		i.image = image
		i.load_image()
		return i
	
	@classmethod
	def from_instance(cls, instance: Self):
		i = cls()
		i.image = instance.image
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
	

	def run_function(self, func: Callable[[Self], Self], **kwargs):
		"""
			Run a function that exposes the instance of the class
		"""
		func(self, **kwargs)
		return self

	def save(self, output_path: str, optimize=False):
		"""Save the image to 'output_path'"""
		self.image.save(output_path, optimize=optimize)
		return self
	
	def show(self):
		plt.imshow(self.image)
		plt.show()
		return self


	# CONDITIONAL
	def run_if_condition(self, condition, function: Callable[[Self], Self]):
		if condition:
			function(self)
		return self

	# ITERATIVE
	def run_iterations(self, parameter_list: List[Any], function: Callable[[Self], Self]):
		for parameter in parameter_list:
			function(self, parameter)
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
	

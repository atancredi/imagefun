from enum import Enum
from math import sqrt
import numpy as np
from json import JSONEncoder
# from typing import Self
from logging import Logger

from PIL import Image, ImageStat, ImageOps


class ColorSpaces(Enum):
	RGB = "RBG"

class MathEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.float32) or isinstance(o, np.float64):
            return float(o)
        return o.__dict__


brightness_magic_values = (0.299, 0.587, 0.114)
class Imagefun:
	image: Image.Image
	logger: Logger

	path: str

	image_palette_normalized: np.ndarray
	image_palette_colors: list
	image_palette_with_percentages: list

	def __init__(self, logger: Logger = None):
		self.filters = []
		self.image = None
		self.logger = logger


	@classmethod
	def from_file(cls, path, logger: Logger = None):
		i = cls(logger)
		i.image = Image.open(path)
		i.path = path
		# i.load_image()
		if i.logger:
			i.logger.debug("[IMG_LOADED] Loaded image from file.", extra={"path": path})
		return i


	@classmethod
	def from_image(cls, image: Image.Image, logger: Logger = None):
		i = cls(logger)
		i.image = image
		# i.load_image()
		if i.logger:
			i.logger.debug("[IMG_LOADED] Loaded image from PIL Image.")
		return i


	@classmethod
	# def from_instance(cls, instance: Self):
	def from_instance(cls, instance, logger: Logger = None):
		i = cls(logger)
		i.image = instance.image
		if logger is not None:
			i.logger = logger
		else:
			i.logger = instance.logger
		# i.load_image()
		if i.logger:
			i.logger.debug("[IMG_LOADED] Loaded image from Imagefun instance.")
		return i


	# Run arbitrary function if condition is met
	def run_if_condition(self, condition, function):
		if condition:
			function(self)
		return self
	

	def save(self, output_path: str, optimize=False):
		"""Save the image to 'output_path'"""
		self.image.save(output_path, optimize=optimize)
		return self
	

	# PROPERTIES
	@property
	def size(self):
		# (width, height)
		return self.image.size
	

	@property
	def brightness(self):
		stat = ImageStat.Stat(self.image)
		channels = stat.mean
		return sqrt(
			sum([x * (y**2) for x, y in zip(brightness_magic_values, channels)], 0)
		)


	# RESIZE
	def _resize(self, new_size):
		self.image = self.image.resize(new_size, Image.Resampling.LANCZOS)
		# self.load_image()

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


	# UTILITIES
	@classmethod
	def invert(i):
		i.image = ImageOps.invert(i.image.convert("L"))
		return i

from enum import Enum
from math import sqrt
import numpy as np
from typing import Literal, Optional, List
# from typing import Self
from logging import Logger
import matplotlib.pyplot as plt

from PIL import Image, ImageStat, ImageOps

from .pipeline_builder import PipelineBuilder
from .functions.palette_generation import generate_palette, generate_palette_old
# from .functions.dithering_halftone import halftone_dither, density_halftone
from .functions.dithering import bayer, floyd_steinberg, diffusion

class ColorSpaces(Enum):
	RGB = "RBG"


class ImagefunBase:
	image: Image.Image
	logger: Logger

	@classmethod
	def from_file(cls, path):
		i = cls()
		i.image = Image.open(path)
		i.path = path
		# i.load_image()
		if i.logger:
			i.logger.debug("[IMG_LOADED] Loaded image from file.", extra={"path": path})
		return i

	@classmethod
	def from_image(cls, image: Image.Image):
		i = cls()
		i.image = image
		# i.load_image()
		if i.logger:
			i.logger.debug("[IMG_LOADED] Loaded image from PIL Image.")
		return i
	
	@classmethod
	# def from_instance(cls, instance: Self):
	def from_instance(cls, instance):
		i = cls()
		i.image = instance.image
		i.logger = instance.logger
		# i.load_image()
		if i.logger:
			i.logger.debug("[IMG_LOADED] Loaded image from Imagefun instance.")
		return i


class ImagefunOps(ImagefunBase):

	# UTILITIES
	@classmethod
	def invert(i):
		i.image = ImageOps.invert(i.image.convert("L"))
		return i


brightness_magic_values = (0.299, 0.587, 0.114)
class Imagefun(ImagefunBase, PipelineBuilder):
	path: str

	image_palette_normalized: np.ndarray
	image_palette_colors: list
	image_palette_with_percentages: list

	def __init__(self):
		self.filters = []
		self.image = None
		self.logger = None

	# IMAGE FUNCTIONS
	def run_filter(self, func, **kwargs):
		"""
			Runs a filter function iteratively on the pixels
			func ( (pixel, **kwargs) -> (float, float, float) )
		"""
		image_array = np.array(self.image)
		image_array = np.array(
			[
				[func(image_array[y, x], **kwargs) for x in range(image_array.shape[1])]
				for y in range(image_array.shape[0])
			]
		)
		self.image = Image.fromarray(image_array)
		return self

	def run_manipulation(self, func, **kwargs):
		"""
			Run a function that manipulates the whole image\n
			func ( (image: Image, **kwargs) -> Image )
		"""
		self.image = func(self.image, **kwargs)
		return self

	def save(self, output_path: str, optimize=False):
		"""Save the image to 'output_path'"""
		self.image.save(output_path, optimize=optimize)
		return self
	
	def show(self):
		plt.imshow(self.image)
		plt.show()
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


	# PALETTE FUNCTIONS
	def palette_old(self, num_colors):
		palette = generate_palette_old(self.image, num_colors, logger=self.logger)
		self.image_palette_colors = palette
		self.image_palette_normalized = palette / 255
		return self
	
	def palette(self, num_colors=8, sample_pixels=500000, with_percentages=False):
		palette = generate_palette(self.image, num_colors, sample_pixels=sample_pixels, logger=self.logger, with_percentages=with_percentages)
		# XXX hack
		if with_percentages:
			self.image_palette_with_percentages = palette
		else:
			self.image_palette_colors = palette
			self.image_palette_normalized = palette / 255
		return self


	@staticmethod
	def is_palette_normalized(palette: List[List[float]]): # XXX utility function - refactor it 
		return all([all([(n * 255) < 256 for n in x]) for x in palette ])


	def dithering(
			self,
			mode: Literal["diffusion"] | Literal["fs"] | Literal["bayer"] = "diffusion",
			palette: Optional[List[float]] = None	
		):
		"""
			modes: "diffusion", "fs" (Floyd-Steinberg), "bayer"
		"""
		use_palette = palette if palette != None else self.image_palette_normalized
		use_palette = use_palette if self.is_palette_normalized(use_palette) else use_palette / 255

		match mode:
			case "diffusion":
				self.image = diffusion(self.image, use_palette)
				
			case "fs":
				self.image = floyd_steinberg(self.image, use_palette)
			
			case "bayer":
				self.image = bayer(self.image, use_palette, 4)

		return self


	# TODO test and implement
	def halftone_dither(
			self,
			channel='r',
			grid_size=10,
			dot_scale=1.5,
			background_color=(255, 255, 255),
			dot_color=(0, 0, 0)
		):
		# self.image = halftone_dither(
		#     self.image,
		#     channel,
		#     grid_size,
		#     dot_scale,
		#     background_color,
		#     dot_color)
		# return self
		raise NotImplementedError()

	# TODO test and implement
	def density_halftone(
			self,
			channel='r',
			num_dots=1e6,
			dot_size=1,
			background_color=(255, 255, 255),
			dot_color=(0, 0, 0)
		):
		# self.image = density_halftone(
		#     self.image,
		#     channel,
		#     num_dots,
		#     dot_size,
		#     background_color,
		#     dot_color
		# )
		# return self
		raise NotImplementedError()


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


	# TODO to move
	def get_keys(self, keys):
		data_dict = {}
		for k in keys:
			if type(k) == str:
				data_dict[k] = self.__dict__[k]
		return data_dict


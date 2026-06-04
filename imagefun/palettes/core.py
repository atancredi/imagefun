from typing import Literal, Optional, List
import numpy as np

from ..core import Imagefun
# from .dithering import halftone_dither, density_halftone
from .dithering import bayer, floyd_steinberg, diffusion
from .functions import generate_palette

class Palettes(Imagefun):

	def __init__(self, logger):
		super().__init__(logger)
	
	def palette(self, num_colors=8, sample_pixels=500000, with_percentages=False):
		palette = generate_palette(self.image, num_colors, sample_pixels=sample_pixels, logger=self.logger, with_percentages=with_percentages)
		# XXX hack
		if with_percentages:
			self.image_palette_with_percentages = palette
		else:
			self.image_palette_colors = palette.tolist()
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
		use_palette = use_palette if self.is_palette_normalized(use_palette) else np.asarray(use_palette) / 255

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


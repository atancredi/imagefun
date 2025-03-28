from PIL import Image
import numpy as np
from tqdm import  tqdm

from imagefun import Imagefun

# TODO just for fun, use a class decorator to avoid declaring __init__ (like dataclasses i guess)
class ComboFilters(Imagefun):
    
	def __init__(self, filename, properties):
		super().__init__(filename, properties)

	def saturation(self, amount: int):
		width = self.image.width
		height = self.image.height
		image2 = Image.new("RGB", (width, height))
		newpixel = [0, 0, 0]
		for x in tqdm(range(width), desc="Saturation"):
			for y in tqdm(range(height),leave=False):
				oldpixel = self.image.getpixel((x, y))
				average = sum(oldpixel) // 3
				for i in range(3):
					newpixel[i] = int(
						max(
							min((oldpixel[i] - average) * amount + average, 
								255),
								0))
					image2.putpixel((x, y), tuple(newpixel))
		return self

	def invert_channels(self):
		width = self.image.width
		height = self.image.height
		image3 = Image.new("RGB", (width, height))
		for x in tqdm(range(width), desc="Inverting channels"):
			for y in tqdm(range(height),leave=False):
				oldpixel = self.image.getpixel((x, y))
				newpixel = (oldpixel[2], oldpixel[0], oldpixel[1]) 
				image3.putpixel((x, y), newpixel)
		return self

	def rebalance_levels(self):
		width = self.image.width
		height = self.image.height
		image4 = Image.new("RGB", (width, height))
		for x in tqdm(range(width), desc="Rebalancing levels"):
			for y in tqdm(range(height),leave=False):
				oldpixel = self.image.getpixel((x, y))
				newpixel = (int(min(oldpixel[0] * 1.9, 255)),
					oldpixel[1],
					int(oldpixel[2] * 0.1))
				image4.putpixel((x, y), newpixel)
		self.image = image4
		return self

	def bitwise_or(self):
		image_np = np.array(self.image, dtype=np.uint8)
		for i in tqdm(range(1, image_np.shape[0] - 1), desc="Bitwise OR"):
			image_np[i:-1:5, 1:-1:5] |= 0xE0  # Apply bitwise OR only on valid pixels
			
			# self.image = Image.fromarray(image_np, "RGB") # NOSONAR mi sa che era sbagliato
		
		self.image = Image.fromarray(image_np, "RGB")
		return self

	def bitwise_and(self):
		image_np = np.array(self.image, dtype=np.uint8)
		for i in tqdm(range(1, image_np.shape[0] - 1), desc="Bitwise AND"):
			image_np[i:-1, 1:-1] &= 0xE0  # Apply bitwise OR only on valid pixels
		
			# image5 = Image.fromarray(image_np, "RGB") # NOSONAR mi sa che era sbagliato

		self.image = Image.fromarray(image_np, "RGB")
		return self


import fire
def hello(filename):

	img = ComboFilters(filename)
    
	img.saturation(20)\
	.invert_channels()\
	.rebalance_levels()\
	.bitwise_and()\
	.save()
	# .bitwise_or()\
    

if __name__ == '__main__':
  fire.Fire(hello)
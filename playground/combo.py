from PIL import Image
import numpy as np
from tqdm import  tqdm

def saturation(image: Image, amount: int):
    width = image.width
    height = image.height
    image2 = Image.new("RGB", (width, height))
    newpixel = [0, 0, 0]
    for x in tqdm(range(width), desc="Saturation"):
        for y in tqdm(range(height),leave=False):
            oldpixel = image.getpixel((x, y))
            average = sum(oldpixel) // 3
            for i in range(3):
                newpixel[i] = int(
                    max(
                        min((oldpixel[i] - average) * amount + average, 
                            255),
                            0))
                image2.putpixel((x, y), tuple(newpixel))
    return image2

def invert_channels(image: Image):
    width = image.width
    height = image.height
    image3 = Image.new("RGB", (width, height))
    for x in tqdm(range(width), desc="Inverting channels"):
        for y in tqdm(range(height),leave=False):
            oldpixel = image.getpixel((x, y))
            newpixel = (oldpixel[2], oldpixel[0], oldpixel[1]) 
            image3.putpixel((x, y), newpixel)
    return image3


def rebalance_levels(image: Image):
    width = image.width
    height = image.height
    image4 = Image.new("RGB", (width, height))
    for x in tqdm(range(width), desc="Rebalancing levels"):
        for y in tqdm(range(height),leave=False):
            oldpixel = image.getpixel((x, y))
            newpixel = (int(min(oldpixel[0] * 1.9, 255)),
        		oldpixel[1],
            	int(oldpixel[2] * 0.1))
            image4.putpixel((x, y), newpixel)
    return image4

def bitwise_or(image: Image):
    image_np = np.array(image, dtype=np.uint8)
    for i in tqdm(range(1, image_np.shape[0] - 1), desc="Bitwise OR"):
        image_np[i:-1:5, 1:-1:5] |= 0xE0  # Apply bitwise OR only on valid pixels
        image5 = Image.fromarray(image_np, "RGB")
    return image5

def bitwise_and(image: Image):
    image_np = np.array(image, dtype=np.uint8)
    for i in tqdm(range(1, image_np.shape[0] - 1), desc="Bitwise AND"):
        image_np[i:-1, 1:-1] &= 0xE0  # Apply bitwise OR only on valid pixels
        image5 = Image.fromarray(image_np, "RGB")
    return image5

import fire
def hello(filename):
    image=Image.open(filename)
    height = int(image.height * 1440 / image.width)
    width = 1440
    image = image.resize((width, height))
    
    image2 = saturation(image, 20)
    image3 = invert_channels(image2)
    image4 = rebalance_levels(image3)
    image5 = bitwise_and(image4)
    # image5 = bitwise_or(image4)
    _ = filename.split(".")
    f =  _[0] + "_" + 'edit' + "." + _[1]
    image5.save(f)
    

if __name__ == '__main__':
  fire.Fire(hello)
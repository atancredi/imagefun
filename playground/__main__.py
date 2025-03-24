from PIL import Image
import fire
from tqdm import tqdm
import numpy as np

def filter_bitwise_or(path):
    image = Image.open(path).convert("RGB")
    image_np = np.array(image, dtype=np.uint8)
    
    # # Apply bitwise AND using NumPy operations
    # image_np[1:-1, 1:-1] &= 0xE0  

    #  # Iterate over rows with tqdm for progress tracking
    # for i in tqdm(range(1, image_np.shape[0] - 1), desc="Processing Rows"):
    #     image_np[i, 1:-1] &= 0xE0  # Apply bitwise AND only on valid pixels

    # height, width, _ = image_np.shape
    # for y in tqdm(range(1, height - 1, 2), desc="Processing Rows"):
    #     for x in range(1, width - 1, 2):
    #         image_np[y, x] &= 0xE0  # Apply bitwise AND to every other pixel



     # Iterate over rows with tqdm for progress tracking
    for i in tqdm(range(1, image_np.shape[0] - 1), desc="Processing Rows"):
        image_np[i:-1:2, 1:-1:2] |= 0xE0  # Apply bitwise OR only on valid pixels
    
    
    # Convert back to image
    image2 = Image.fromarray(image_np, "RGB")
    return image2

def filter_bitwise_and(path):
    image = Image.open(path).convert("RGB")
    image_np = np.array(image, dtype=np.uint8)
    
    # # Apply bitwise AND using NumPy operations
    # image_np[1:-1, 1:-1] &= 0xE0  

     # Iterate over rows with tqdm for progress tracking
    for i in tqdm(range(1, image_np.shape[0] - 1), desc="Processing Rows"):
        image_np[i, 1:-1] &= 0xE0  # Apply bitwise AND only on valid pixels

    # height, width, _ = image_np.shape
    # for y in tqdm(range(1, height - 1, 2), desc="Processing Rows"):
    #     for x in range(1, width - 1, 2):
    #         image_np[y, x] &= 0xE0  # Apply bitwise AND to every other pixel

    # Convert back to image
    image2 = Image.fromarray(image_np, "RGB")
    return image2

def hello(filename, filter = "bitwise_and"):
    match filter:
        case 'bitwise_and':
            image = filter_bitwise_and(filename)
            _ = filename.split(".")
            f =  _[0] + "_" + filter + "." + _[1]
            image.save(f)
        case 'bitwise_or':
            image = filter_bitwise_or(filename)
            _ = filename.split(".")
            f =  _[0] + "_" + filter + "." + _[1]
            image.save(f)
if __name__ == '__main__':
  fire.Fire(hello)
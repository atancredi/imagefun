# resize all images in a folder
import sys
sys.path.insert(0,'../imagefun/')
import os
from pathlib import Path
from tqdm import tqdm

from imagefun.resize import Resize

SOURCE_FOLDER = ""
OUTPUT_FOLDER = ""
FACTOR = 0.5

def files_in_folder(directory, extensions):
    extensions = [ext.lower() for ext in extensions]
    for root, _, files in os.walk(directory):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in extensions:
                yield file


if __name__ == "__main__":

    def convert_rgb(r: Resize):
        r.image.convert("RGB")

    # save images in different folders by orientation
    os.makedirs(Path(OUTPUT_FOLDER) / "landscape", exist_ok=True)
    os.makedirs(Path(OUTPUT_FOLDER) / "portrait", exist_ok=True)
    def save_by_orientation(r: Resize):
        if r.size[0] > r.size[1]:
            r.save(Path(OUTPUT_FOLDER) / "landscape" / (os.path.splitext(image)[0]+".png"))
        else:
            r.save(Path(OUTPUT_FOLDER) / "portrait" / (os.path.splitext(image)[0]+".png"))
    
    for image in tqdm(files_in_folder(SOURCE_FOLDER, [".cr2"])):
        r = Resize()\
            .from_file(os.path.join(SOURCE_FOLDER, image))\
            .resize_by_factor(FACTOR)\
            .run_function(convert_rgb)\
            .run_function(save_by_orientation)
            # .save(os.path.join(OUTPUT_FOLDER, os.path.splitext(image)[0]+".png"))

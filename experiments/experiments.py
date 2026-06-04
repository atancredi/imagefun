from os import listdir, makedirs
from os.path import join, isdir, isfile, splitext, basename
from typing import Tuple
from json import load
from pathlib import Path

IMAGE_EXTENSIONS = [".png", ".jpg", ".jpeg"]
EXPERIMENT_FILE = "experiment.json"

def list_dir(root_path):
    for path in listdir(root_path):
        full_path = join(root_path, path)
        yield (full_path, isfile(full_path), isdir(full_path))


def load_images_and_config(experiment_folder, image_extensions, config_file_extension):
    # check if images and the json are in the folder
    config_file = None
    image_files = []
    for full_path, is_file, is_dir in list_dir(experiment_folder):
        if is_file:
            _ext: Tuple[str, str] = splitext(full_path)
            if _ext[1].lower() in image_extensions:
                image_files.append(full_path)
            if basename(full_path) == EXPERIMENT_FILE:
                    config_file = full_path

    if config_file == None:
        raise FileNotFoundError("No .json file in folder "+experiment_folder)
    if len(image_files) == 0:
        raise FileNotFoundError("No images in folder "+experiment_folder)
    
    return config_file, image_files


class ImagefunExperimentManager:

    def __init__(self,
                 experiment_folder: str | Path,
                 results_folder = "results"):
        self.experiment_folder = Path(experiment_folder)
        self.results_folder = results_folder

        # make results folder
        makedirs(Path(experiment_folder) / results_folder, exist_ok=True)

        self.config_file = None
        self.image_files = []


    def load_experiments(self):
        self.experiments = load(open(self.config_file,"r"))


    def prepare_images(self):
        config_file, image_files = load_images_and_config(self.experiment_folder, IMAGE_EXTENSIONS, EXPERIMENT_FILE)

        self.config_file = config_file
        self.image_files = image_files

        # create results folder
        makedirs(self.experiment_folder / self.results_folder, exist_ok=True)

    @property
    def results_folder_path(self):
        return self.experiment_folder / self.results_folder

    def __enter__(self):
        self.prepare_images()
        self.load_experiments()
        return self


    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
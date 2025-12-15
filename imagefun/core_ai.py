# Class with AI functions
# Separated from the base Imagefun to keep the dependencies light

from .core import Imagefun
from .functions_ai.embeddings import palette_vector, image_vector, full_image_vector, setup_model


class ImagefunAI(Imagefun):

    image_palette_vector: list
    image_vector: list
    image_combined_vector: list

    def __init__(self):
        super().__init__()
        self.image_palette_vector = None
        self.image_vector = None
        self.image_combined_vector = None
        self.model = None
        self.device = None

    def setup_model(self):
        model, device = setup_model()
        self.model = model
        self.device = device
        return self

    def compute_palette_vector(self):
        # if self.image_palette_normalized == None:
        #     raise ValueError("Must calculate palette first.")
        self.image_palette_vector = palette_vector(self.image_palette_normalized)
        return self

    def compute_image_vector(self):
        # if self.model == None:
        #     raise ValueError("Must setup model first.")
        self.image_vector = image_vector(self.image, self.model)
        return self

    def compute_combined_image_vector(self, alpha_mix = 0.7):
        # if self.image_vector == None:
        #     raise ValueError("Must calculate image vector first.")
        # if self.image_palette_vector == None:
        #     raise ValueError("Must calculate palette vector first.")
        self.image_combined_vector = full_image_vector(self.image_vector, self.image_palette_vector, alpha_mix)
        return self
    

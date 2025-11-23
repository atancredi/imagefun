from imagefun import Imagefun, ImageProperties
from imagefun.manipulations.matrix_conversion import matrix_conversion, edge_detect_pil
import numpy as np

# from imagefun.manipulations.image_enhance import image_enhance
# from PIL import ImageEnhance

def edge_kernel(alpha=1.0):
    """
    alpha controls edge sharpness (higher = stronger contrast)
    """
    return [
        [0, -alpha, 0],
        [-alpha, 4 * alpha, -alpha],
        [0, -alpha, 0]
    ]


if __name__ == "__main__":

    props = ImageProperties(width=720 * 3)

    f = (
        Imagefun(properties=props)
        .from_file("_testimages/test_exwide.jpeg")
        .print_brightness()
        .run_manipulation(
            # matrix_conversion,
            edge_detect_pil,
            matrix=edge_kernel(1)
            # matrix=[[42, 5, 66], [0, 0, 0], [-40, -40, -40], [-55, -55, -55]],
            # matrix=[[42, 5, 0], [0, 0, 0], [-40, -40, -0], [-55, -55, -0]], #Cool!
            # matrix=[[0, 5, 66], [0, 0, 0], [-0, -40, -40], [-0, -55, -55]],
        )
        # .run_manipulation(
        #     image_enhance,
        #     enhancer=ImageEnhance.Contrast,
        #     value=2.0
        # )
        .print_brightness()
        .save("_results/test_exwide_res.jpg")
    )

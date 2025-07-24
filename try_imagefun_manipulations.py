from imagefun import Imagefun, ImageProperties
from imagefun.manipulations.matrix_conversion import matrix_conversion

# from imagefun.manipulations.image_enhance import image_enhance
# from PIL import ImageEnhance


if __name__ == "__main__":

    props = ImageProperties(width=720 * 3)

    f = (
        Imagefun(properties=props)
        .from_file("_testimages/test_ale.jpg")
        .print_brightness()
        .run_manipulation(
            matrix_conversion,
            # matrix=[[42, 5, 66], [0, 0, 0], [-40, -40, -40], [-55, -55, -55]],
            matrix=[[42, 5, 0], [0, 0, 0], [-40, -40, -0], [-55, -55, -0]],
            # matrix=[[0, 5, 66], [0, 0, 0], [-0, -40, -40], [-0, -55, -55]],
        )
        # .run_manipulation(
        #     image_enhance,
        #     enhancer=ImageEnhance.Contrast,
        #     value=2.0
        # )
        .print_brightness()
        .save("_results/test_ale_res.jpg")
    )

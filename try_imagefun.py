from PIL import Image, ImageEnhance

from imagefun import Imagefun, ImageProperties
from imagefun.filters.rgb import cross_bwand, make_darker
from imagefun.manipulations.matrix_conversion import matrix_conversion

from imagefun.manipulations.image_enhance import image_enhance

if __name__ == "__main__":

    props = ImageProperties(width=720*3)

    # f = (
    #     Imagefun(properties=props)
    #     .from_file("testimages/test_ale.jpg")
    #     .add_filter(cross_bwand)
    #     .add_filter(make_darker)
    #     .process()
    #     .brightness()
    #     .add_filter(make_darker)
    #     .process()
    #     .brightness()
    #     .save("results/test_ale_joep_imfv2.jpg")
    # )

    f = (
        Imagefun(properties=props)
        .from_file("testimages/test_ale.jpg")
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
        # .save("results/test_ale_coolmatrix1.jpg")
        # .save("results/test_ale_enh.jpg")
    )

    f_orig = (
        Imagefun(properties=props).from_file("testimages/test_ale.jpg")
    )

    # Alpha blending: 0.0 = only dry, 1.0 = only wet
    alpha = 0.3  # 40% wet, 60% dry

    # Blend them
    blended = Image.blend(f_orig.image, f.image, alpha)

    # Save result
    blended.save("blended.jpg")

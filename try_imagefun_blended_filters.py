from PIL import Image

from imagefun import Imagefun, ImageProperties
from imagefun.filters.rgb import cross_bwand, make_darker

if __name__ == "__main__":

    props = ImageProperties(width=720*3)

    f = (
        Imagefun(properties=props)
        .from_file("_testimages/test_ale.jpg")
        .add_pixel_filter(cross_bwand)
        .add_pixel_filter(make_darker)
        .process_pixels()
    )

    f_orig = (
        Imagefun(properties=props).from_file("_testimages/test_ale.jpg")
    )

    # Alpha blending: 0.0 = only dry, 1.0 = only wet
    alpha = 0.3  # 40% wet, 60% dry

    # Blend them
    blended = Image.blend(f_orig.image, f.image, alpha)

    # Save result
    blended.save("_results/test_ale_blended.jpg")

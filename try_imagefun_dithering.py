from imagefun import ImageProperties
from imagefun.dithering import Dithering

if __name__ == "__main__":

    props = ImageProperties()


    f = (
        Dithering(properties=props)
        .from_file("_testimages/test_ale.jpg")
        .compress_for_web(16)
        .save("_results/test_ale_webDither_16.png")
    )

    f = (
        Dithering(properties=props)
        .from_file("_testimages/test_ale.jpg")
        .halftone_dither(channel='r', grid_size=8, dot_scale=1.6)
        .save("_results/test_ale_halftone_r.png")
    )

    f = (
        Dithering(properties=props)
        .from_file("_testimages/test_ale.jpg")
        .halftone_dither(channel='b', grid_size=12, dot_scale=1.2)
        .save("_results/test_ale_halftone_b.png")
    )

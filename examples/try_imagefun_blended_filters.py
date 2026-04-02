import sys
sys.path.insert(0,'../imagefun/')

from PIL import Image

from imagefun import Imagefun
from imagefun.filters.rgb import cross_bwand, make_darker

if __name__ == "__main__":

    f = (
        Imagefun
        .from_file("_testimages/test_ale.jpg")
        .run_filter(cross_bwand)
        .run_filter(make_darker)
    )

    f_orig = (
        Imagefun.from_file("_testimages/test_ale.jpg")
    )

    # Alpha blending: 0.0 = only dry, 1.0 = only wet
    alpha = 0.3  # 40% wet, 60% dry

    # Blend them
    blended = Image.blend(f_orig.image, f.image, alpha)

    # Save result
    blended.save("_results/test_ale_blended.jpg")

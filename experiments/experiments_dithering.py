import sys
sys.path.insert(0,'../imagefun/')

from imagefun import ImageProperties, Dithering, get_logger
from imagefun.experiments import ImagefunExperimentManager

logger = get_logger("dithering")

with ImagefunExperimentManager("_test_dither") as exps:
    for e in exps.experiments:

        props = ImageProperties()

        file = e.get("file")
        n_colors = e.get("n_colors", [])
        palettes = e.get("palettes", [])

        p = exps.experiment_folder / file

        for n_color in n_colors:
            o = exps.results_folder_path / (file.split(".")[0] + "_dithered_" + str(n_color) + ".png")
            o_p = exps.results_folder_path / (file.split(".")[0] + "__dithered_palette_" + str(n_color) + ".png")
            f = (
                Dithering(properties=props, logger=logger)
                .from_file(p)
                .palette_3(n_color)
                .diffusion()
                # .make_indexed_png_2()
                .plot_palette(o_p)
                .save(o)
            )

        for palette in palettes:
            fname = (file.split(".")[0] + f"_dithered_{palette['tag']}")
            if palette.get("invert"): fname += "_inverted"
            fname += ".png"
            o = exps.results_folder_path / fname

            f = (
                Dithering(properties=props, logger=logger)
                .from_file(p)
                .set_palette(palette["colors"])
                .diffusion()
                .conditional(
                    palette.get("invert", False),
                    lambda d: d.invert()
                )
                .save(o)
            )

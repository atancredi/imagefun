import sys
sys.path.insert(0,'../imagefun/')
from json import dump
from collections import defaultdict

from imagefun import MathEncoder
from imagefun.stacklogger import get_logger
from imagefun.dithering import Dithering
from imagefun.experiments import ImagefunExperimentManager


logger = get_logger("dithering")

with ImagefunExperimentManager("_test_dither") as exps:
    report = defaultdict(dict)
    for e in exps.experiments:

        file = e.get("file")
        n_colors = e.get("n_colors", [])
        palettes = e.get("palettes", [])

        p = exps.experiment_folder / file

        for n_color in n_colors:
            o = exps.results_folder_path / (file.split(".")[0] + "_dithered_" + str(n_color) + ".png")
            o_p = exps.results_folder_path / (file.split(".")[0] + "__dithered_palette_" + str(n_color) + ".png")
            f = (
                Dithering()
                .set_logger(logger)
                .from_file(p)
                .palette_3(n_color)
                .diffusion()
                # .make_indexed_png_2()
                .plot_palette(o_p)
                .save(o)
            )
            report[file][n_color] = f.image_palette_normalized * 255

        for palette in palettes:
            fname = (file.split(".")[0] + f"_dithered_{palette['tag']}")
            if palette.get("invert"): fname += "_inverted"
            fname += ".png"
            o = exps.results_folder_path / fname

            f = (
                Dithering()
                .set_logger(logger)
                .from_file(p)
                .set_palette(palette["colors"])
                .diffusion()
                .run_if_condition(
                    palette.get("invert", False),
                    lambda d: d.invert()
                )
                .save(o)
            )

    dump(report, open("_test_dither/results/experiment_results.json", "w+"), cls=MathEncoder)

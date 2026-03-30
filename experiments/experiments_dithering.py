import sys
sys.path.insert(0,'../imagefun/')
from json import dump
from collections import defaultdict
from pathlib import Path

from imagefun import Imagefun, ImagefunOps, MathEncoder, get_logger
from experiments import ImagefunExperimentManager


logger = get_logger("dithering")



import argparse
p = argparse.ArgumentParser()
p.add_argument("folder")
args = p.parse_args()

# with ImagefunExperimentManager("_test_dither") as exps:
with ImagefunExperimentManager(args.folder) as exps:
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
                Imagefun()
                .set_logger(logger)
                .from_file(p)
                .resize_linked(1250)
                # .palette_old(n_color)
                .palette(n_color)
                .dithering()
                # .run_function(plot_palette, output_name=o_p)
                .save(o)
            )
            report[file][n_color] = f.image_palette_normalized * 255

        for palette in palettes:
            fname = (file.split(".")[0] + f"_dithered_{palette['tag']}")
            if palette.get("invert"): fname += "_inverted"
            fname += ".png"
            o = exps.results_folder_path / fname

            f = (
                Imagefun()
                .set_logger(logger)
                .from_file(p)
                .dithering(palette=palette["colors"])
                .run_if_condition(
                    palette.get("invert", False),
                    ImagefunOps.invert
                )
                .save(o)
            )

    report_path = Path(args.folder) / "results" / "experiment_results.json"
    dump(report, open(report_path, "w+"), cls=MathEncoder)

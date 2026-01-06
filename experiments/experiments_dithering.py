import sys
sys.path.insert(0,'../imagefun/')
from json import dump
from collections import defaultdict
import numpy as np
from PIL import ImageOps
import matplotlib.pyplot as plt
from pathlib import Path

from imagefun import Imagefun, MathEncoder, get_logger
from experiments import ImagefunExperimentManager


logger = get_logger("dithering")


def set_palette(i: Imagefun, palette):
    # accepts both normalized and not-normalized paletted
    # if a color channel is > 1 it is not normalized (this is maybe a bit weak)
    is_norm = True
    for color in palette:
        for channel in color:
            if channel > 1:
                is_norm = False
                break

    palette = np.asarray(palette)

    if not is_norm:
        palette = palette / 255

    i.image_palette_normalized = palette
    return i


def plot_palette(i: Imagefun, output_name: str = None):
    palette_normalized = i.image_palette_normalized
    _, ax = plt.subplots(figsize=(len(palette_normalized), 1), dpi=80)
    ax.imshow([palette_normalized], aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    plt.title("Extracted Color Palette")
    
    if output_name != None:
        plt.savefig(output_name)
    else:
        plt.show()

    return i


def invert_image(i: Imagefun):
    i.image = ImageOps.invert(i.image.convert("L"))
    return i



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
                # .set_palette(palette["colors"])
                .run_function( # TODO should test it before pushing !
                    set_palette,
                    palette=palette["colors"]
                )
                .dithering()
                .run_if_condition(
                    palette.get("invert", False),
                    invert_image
                )
                .save(o)
            )

    report_path = Path(args.folder) / "results" / "experiment_results.json"
    dump(report, open(report_path, "w+"), cls=MathEncoder)

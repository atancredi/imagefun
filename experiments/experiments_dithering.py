import sys
sys.path.insert(0,'../imagefun/')
import os
from json import dump
from collections import defaultdict
from pathlib import Path

from imagefun import Imagefun, ImagefunOps, MathEncoder, get_logger
from experiments import ImagefunExperimentManager


logger = get_logger("dithering")


def main(folder: str):

    with ImagefunExperimentManager(folder) as exps:
        report = defaultdict(dict)
        for e in exps.experiments:

            file = e.get("file")
            n_colors = e.get("n_colors", [])
            palettes = e.get("palettes", [])

            file_name = os.path.basename(file)

            for n_color in n_colors:
                f = (
                    Imagefun()
                    .set_logger(logger)
                    .from_file(
                        exps.experiment_folder / file
                    )
                    .palette(n_color)
                    .dithering()
                    .save(
                        exps.results_folder_path / f"{file_name}_dithered_{n_color}.png"
                    )
                )
                report[file][n_color] = f.image_palette_normalized * 255

            for palette in palettes:

                invert = palette.get("invert", False)
                f = (
                    Imagefun()
                    .set_logger(logger)
                    .from_file(
                        exps.experiment_folder / file
                    )
                    .dithering(palette=palette["colors"])
                    .run_if_condition(
                        invert,
                        ImagefunOps.invert
                    )
                    .save(
                        exps.results_folder_path / f"{file_name}_dithered_{palette['tag']}{"_inverted" if invert else ""}.png"
                    )
                )

        report_path = exps.results_folder_path / "experiment_results.json"
        dump(report, open(report_path, "w+"), cls=MathEncoder)


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
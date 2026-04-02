import sys
sys.path.insert(0,'../imagefun/')
import os
from collections import defaultdict
from json import dump

from imagefun import MathEncoder, get_logger
from imagefun.modules import Palettes
from experiments import ImagefunExperimentManager

logger = get_logger("dithering")

def main(folder: str, size = 1250):

    with ImagefunExperimentManager(folder, results_folder="resized_dither_results") as exps:
        report = defaultdict(dict)
        for e in exps.experiments:

            file: str = e.get("file")
            n_colors = e.get("n_colors", [])

            file_name = os.path.basename(file)

            # resize it
            resized_path = exps.results_folder_path / (f"{file_name}_resized_{size}.png")
            f = (
                Palettes
                .from_file(
                    exps.experiment_folder / file,
                    logger
                )
                .resize_linked(1250)\
                .save(resized_path)
            )

            # dither the resized image
            for n_color in n_colors:
                f = (
                    Palettes
                    .from_file(
                        resized_path,
                        logger
                    )
                    .palette(n_color)
                    .dithering()
                    .save(
                        exps.results_folder_path / f"{file_name}_resized_{size}_dithered_{n_color}.png"
                    )
                )
                report[file][n_color] = f.image_palette_normalized * 255


        dump(report, open(exps.results_folder_path / "experiment_results.json", "w+"), cls=MathEncoder)


if __name__ == "__main__":
    from fire import Fire
    Fire(main)
import os
from json import dump, JSONEncoder
import numpy as np
from collections import defaultdict

import sys

sys.path.insert(0, "../imagefun/")

from imagefun import ImagefunLoader
from experiments import ImagefunExperimentManager


class MathEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.float32) or isinstance(o, np.float64):
            return float(o)
        return o.__dict__


def main(folder: str, size=1250):

    with ImagefunExperimentManager(
        folder, results_folder="resized_dither_results"
    ) as exps:
        report = defaultdict(dict)
        for e in exps.experiments:

            file: str = e.get("file")
            n_colors = e.get("n_colors", [])
            palettes = e.get("palettes", [])

            file_name = os.path.basename(file)

            # resize it
            resized_path = exps.results_folder_path / (
                f"{file_name}_resized_{size}.png"
            )
            f = (
                ImagefunLoader.from_file(exps.experiment_folder / file)
                .resize_linked(1250)
                .save(resized_path)
            )

            # dither the resized image
            for n_color in n_colors:
                f = (
                    ImagefunLoader.from_file(resized_path)
                    .get_palette(n_color)
                    .dithering()
                    .save(
                        exps.results_folder_path
                        / f"{file_name}_resized_{size}_dithered_{n_color}.png"
                    )
                )
                report[file][n_color] = f.palette_colors

            for i, palette in enumerate(palettes):

                f = (
                    ImagefunLoader.from_file(resized_path)
                    .dithering(palette=palette)
                    .save(
                        exps.results_folder_path
                        / f"{file_name}_resized_{size}_dithered_palette{i}.png"
                    )
                )

        dump(
            report,
            open(exps.results_folder_path / "experiment_results.json", "w+"),
            cls=MathEncoder,
        )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)

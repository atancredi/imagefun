import os
from json import dump
from pathlib import Path
from json import JSONEncoder
import numpy as np
from collections import defaultdict

import sys

sys.path.insert(0, "../imagefun/")
from imagefun import ImagefunLoader


class MathEncoder(JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.float32) or isinstance(o, np.float64):
            return float(o)
        return o.__dict__


def main(folder_path, size=1250):
    report = defaultdict(dict)

    results_folder_path = Path(folder_path) / "pipeline_results"

    os.makedirs(results_folder_path / "resized_dithered", exist_ok=True)
    os.makedirs(results_folder_path / "resized", exist_ok=True)

    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if os.path.isfile(full_path) and file.lower().endswith(
            (".png", ".jpg", ".jpeg")
        ):

            file: str = full_path
            n_colors = [2, 4, 8, 16]

            file_name = os.path.splitext(os.path.basename(file))[0]

            # resize it
            resized_path = (
                results_folder_path / "resized" / (f"{file_name}_resized_{size}.png")
            )
            f = (
                ImagefunLoader.from_file(full_path)
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
                        results_folder_path
                        / "resized_dithered"
                        / f"{file_name}_resized_{size}_dithered_{n_color}.png"
                    )
                )
                report[file][n_color] = f.image_palette_normalized * 255

        dump(
            report,
            open(results_folder_path / "pipeline_results.json", "w+"),
            cls=MathEncoder,
        )


if __name__ == "__main__":
    from fire import Fire

    Fire(main)

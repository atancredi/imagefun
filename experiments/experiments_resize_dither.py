import sys
sys.path.insert(0,'../imagefun/')
from collections import defaultdict
from json import dump

from imagefun import Imagefun, MathEncoder
from imagefun.dithering import Dithering
from imagefun.resize import Resize
from imagefun.experiments import ImagefunExperimentManager
from imagefun.stacklogger import get_logger

logger = get_logger("dithering")

with ImagefunExperimentManager("_test_dither", results_folder="resized_dither_results") as exps:
    for e in exps.experiments:

        file = e.get("file")
        n_colors = e.get("n_colors", [])
        palettes = e.get("palettes", [])

        report = defaultdict(dict)
        target_larger_side = 1250
        image_path = exps.experiment_folder / file
        resized_image_path = exps.results_folder_path / (file.split(".")[0] + f"_resized_{target_larger_side}.png")
        def get_dithered_file_path(file_name, n_color):
            return exps.results_folder_path / (file_name.split(".")[0] + f"_resized_{target_larger_side}_dithered_{n_color}.png")
        def save_value_to_file_report(key, value):
            report[file][key] = value

        Imagefun\
        .from_instance(
            Resize()
            .set_logger(logger)
            .from_file(image_path)
            .resize_linked(target_larger_side)
            .save(resized_image_path)
        )\
        .run_iterations(
            n_colors,
            lambda instance, n_color:
                (
                    Dithering.from_instance(instance)
                    .palette_3(n_color)
                    .diffusion()
                    .run_function(
                        lambda instance: save_value_to_file_report(n_color, instance.image_palette_normalized * 255)
                    )
                    .save(get_dithered_file_path(file, n_color))
                )
        )
        
        dump(report, open("_test_dither/resized_dither_results/experiment_results.json", "w+"), cls=MathEncoder)


import sys
sys.path.insert(0,'../imagefun/')
from collections import defaultdict
from json import dump

from imagefun import Imagefun, MathEncoder, get_logger
from experiments import ImagefunExperimentManager

logger = get_logger("dithering")

with ImagefunExperimentManager("_test_dither", results_folder="resized_dither_results") as exps:
    report = defaultdict(dict)
    for e in exps.experiments:

        file: str = e.get("file")
        n_colors = e.get("n_colors", [])
        palettes = e.get("palettes", [])

        target_larger_side = 1250
        image_path = exps.experiment_folder / file
        resized_image_path = exps.results_folder_path / (file.split(".")[0] + f"_resized_{target_larger_side}.png")
        def get_dithered_file_path(file_name: str, n_color):
            return exps.results_folder_path / (file_name.split(".")[0] + f"_resized_{target_larger_side}_dithered_{n_color}.png")
        def save_value_to_file_report(key, value):
            report[file][key] = value

        Imagefun\
        .from_file(image_path)\
        .set_logger(logger)\
        .resize_linked(target_larger_side)\
        .save(resized_image_path)\
        .run_iterations(
            n_colors,
            lambda instance, n_color:
                (
                    Imagefun.from_instance(instance)
                    .palette_old(n_color)
                    .dithering()
                    .run_function(
                        lambda instance: save_value_to_file_report(n_color, instance.image_palette_normalized * 255)
                    )
                    .save(get_dithered_file_path(file, n_color))
                )
        )
        
    dump(report, open("_test_dither/resized_dither_results/experiment_results.json", "w+"), cls=MathEncoder)


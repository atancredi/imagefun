import sys
sys.path.insert(0,'../imagefun/')

from imagefun import ImageProperties
from imagefun.dithering import Dithering, get_logger
from imagefun.resize import Resize
from imagefun.experiments import ImagefunExperimentManager

logger = get_logger("dithering")

with ImagefunExperimentManager("_test_dither", results_folder="resized_dither_results") as exps:
    for e in exps.experiments:

        report = {}

        props = ImageProperties()

        file = e.get("file")
        n_colors = e.get("n_colors", [])
        palettes = e.get("palettes", [])

        p = exps.experiment_folder / file

        target_larger_side = 1250
        o = exps.results_folder_path / (file.split(".")[0] + f"_resized_{target_larger_side}.png")
        r = (
            Resize(properties=props)
            .from_file(p)
            .resize_image_larger_side(target_larger_side)
            .save(o)
        )
        report['resized'] = r.size

        report['n_colors'] = []
        for n_color in n_colors:
            o = exps.results_folder_path / (file.split(".")[0] + "_dithered_" + str(n_color) + ".png")
            d = (
                Dithering.from_instance(r)
                .palette_3(n_color)
                # .diffusion()
                # .save(o)
            )
            report['n_colors'].append([[int(j) for j in x * 255] for x in d.image_palette_normalized])

        print(report)

        # for n_color in n_colors:
        #     o = exps.results_folder_path / (file.split(".")[0] + "_dithered_" + str(n_color) + ".png")
        #     o_p = exps.results_folder_path / (file.split(".")[0] + "__dithered_palette_" + str(n_color) + ".png")
        #     f = (
        #         Dithering(properties=props, logger=logger)
        #         .from_file(p)
        #         .palette_3(n_color)
        #         .diffusion()
        #         # .make_indexed_png_2()
        #         .plot_palette(o_p)
        #         .save(o)
        #     )

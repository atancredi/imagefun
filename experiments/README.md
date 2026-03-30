# Image experiments


- ### 1. Put all images in a directory under the project's root.

- ### 2. Generate an 'experiment' json file with default settings for every image in the folder with get_experiment_file_from_directory.py with the directory's path as argument:


Remember to run all commands from the project's root.

```bash
python experiments/get_experiment_file_from_directory.py path/to/folder
```

All the settings from the `experiment.json` file, created in the image folder, could be customized.


- ### 3. Run the experiment:
```bash
python experiments/experiments_resize_dither.py path/to/folder
# or
python experiments/experiments_dithering.py path/to/folder
```

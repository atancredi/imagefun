import os
import json
from fire import Fire

def main(folder_path):
    files = []
    for file in os.listdir(folder_path):
        full_path = os.path.join(folder_path, file)
        if os.path.isfile(full_path) and file.lower().endswith((".png", ".jpg")):
            files.append({
                "file": file,
                "n_colors": [2,4,8,16]
            })
    json.dump(files, open(os.path.join(folder_path,"experiment.json"), "w+"))

if __name__ == "__main__":

    Fire(main)
    

    
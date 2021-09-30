import json
import numpy as np


def LoadDataset(filename: str):
    data = json.load(open(filename))
    images = []
    classes = {}
    for k, v in data['images'].items():
        images.append({
            "image_path": k,
            "box": [v_i['box'] for v_i in v],
            "classname": [v_i['classname'] for v_i in v]
        })

    classes = {int(k): v for k, v in data['classnames'].items()}

    return images, classes


if __name__ == "__main__":
    dataset, classes = LoadDataset('')
    print(len(dataset))
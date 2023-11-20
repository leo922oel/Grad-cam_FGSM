import os

import json
import numpy as np
from PIL import Image

def json2img(fname, prefix="./img/"):
    os.makedirs(prefix, exist_ok=True)
    print("Convert to Image ...")
    with open(fname, 'r') as f:
        dataset = json.load(f)
    dataset = np.array(dataset).reshape(len(dataset), 28, -1)

    for idx, data in enumerate(dataset):
        data = ((data+1) / 2.0) * 255
        img = Image.fromarray(data.astype('uint8'))
        name = f'{prefix}idx_{idx+1}.jpg'
        img.save(name)
    print("Finish.")

if __name__ == "__main__":
    json2img('data/71_data.json')
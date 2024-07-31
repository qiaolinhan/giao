# step one is to use labelme label the points, the point pixel values are stored in
# .json files

import json
import numpy as np
import cv2
import pands as pd
from PIL import Image
from labelme import utils

def extract_pixel_values(json_file, image_file):
    # load the json file
    with open(json_file) as f:
        data = json.load(f)

    image = np.array(Image.open(image_file))

    pixel_values = []

    for shape in data['shapes']:
        if shape['shape_type'] == 'point':
            # get the coordinate of the point
            point = shape['points'][0]
            x, y = int(point[0]), int(point[1])
            label = shape['label']

            # get the pixel value at the point
            value = image[x, y]

            # append the label and pixel value
            pixel_values.append([label, x, y, *value])

    return pixel_values
            

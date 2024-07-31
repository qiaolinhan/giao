# step one is to use labelme label the points, the point pixel values are stored in
# .json files

import json
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from labelme import utils
import math

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
            # pixel_values.append([label, x, y, *value])
            pixel_values.append([x, y])

    return pixel_values
#############################         
distance = 6 * 100 * 10 # m --> mm
ir_image_file = './6m/0000_ir.png'
wide_image_file = './6m/0000_wide.png'
zoom_image_file = './6m/0000_zoom.png'
ir_json_file = './6m/0000_ir_6_14.json'
wide_json_file = './6m/0000_wide_6_14.json'
zoom_json_file = './6m/0000_zoom_6_14.json'
#############################

depth = math.sqrt(distance**2 - 1680**2)
print(f'[INFO] :: Acquired the depth: {depth}')

ir_pixel_values = extract_pixel_values(ir_json_file, ir_image_file)
wide_pixel_values = extract_pixel_values(wide_json_file, wide_image_file)
zoom_pixel_values = extract_pixel_values(zoom_json_file, zoom_image_file)

print("[INFO] :: Infrared pixel values:\n", ir_pixel_values)
print("[INFO] :: Wide pixel values:\n", wide_pixel_values)
print("[INFO] :: Zoom pixel values:\n", zoom_pixel_values)

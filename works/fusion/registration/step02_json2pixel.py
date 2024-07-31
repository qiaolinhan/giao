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
distance = 12 * 100 * 10 # m --> mm
ir_image_file = './12m/0000_ir.png'
wide_image_file = './12m/0000_wide.png'
zoom_image_file = './12m/0000_zoom.png'
ir_json_file = './12m/0000_ir_12_12.json'
wide_json_file = './12m/0000_wide_12_12.json'
zoom_json_file = './12m/0000_zoom_12_12.json'
#############################

depth = math.sqrt(distance**2 - 1680**2)
print(f'[INFO] :: Acquired the depth: {depth}')

ir_pixel_values = extract_pixel_values(ir_json_file, ir_image_file)
wide_pixel_values = extract_pixel_values(wide_json_file, wide_image_file)
zoom_pixel_values = extract_pixel_values(zoom_json_file, zoom_image_file)

print("[INFO] :: Infrared pixel values:\n", ir_pixel_values)
print("[INFO] :: Wide pixel values:\n", wide_pixel_values)
print("[INFO] :: Zoom pixel values:\n", zoom_pixel_values)

vi_points_wide = wide_pixel_values
vi_points_zoom = zoom_pixel_values
ir_points = ir_pixel_values
d = depth

#######################
# construct R' and t'
R = []
t = []

# wide and infrared
for (x1, y1), (x2, y2) in zip(vi_points_wide, ir_points):
    R.append([x1, y1, 1, 0, 0, 0, 0, 0, 0, -x2*x1, -x2*y1, -x2])
    R.append([0, 0, 0, x1, y1, 1, 0, 0, 0, -y2*x1, -y2*y1, -y2])
    R.append([0, 0, 0, 0, 0, 0, x1, y1, 1, -1*x1, -1*y1, -1])
    # Removing the row associated with z coordinate transformation
    # and setting R33 = 1, R31 = 0, R32 = 0, t3 = 0
    # This results in R31, R32, t3 not appearing in the linear system

    t.extend([x2-1/d, y2-1/d, 1-1/d])

R = np.array(R)
t = np.array(t)

# solve for the parameters using least squares
params, _, _, _ = np.linalg.lstsq(R, t, rcond=None)

# extract the transformation matrix components
# Manually set R31=R32=0, R33=1
R11, R12, R13, R21, R22, R23, R31, R32, R33, t1, t2, t3 = params  

print("[INFO] :: For wide images and infrared images")
print("[INFO] :: Transform matrix:\n", np.array([[R11, R12, R13], [R21, R22, R23],
                                                 [R31, R32, R33]]))
print("[INFO] :: Translation vector (scaled with d):", np.array([t1, t2, t3]))



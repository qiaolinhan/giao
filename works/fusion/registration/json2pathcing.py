
# step one is to use labelme label the points, the point pixel values are stored in
# .json files

import json
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from labelme import utils
import math
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R

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
# distance, 6m --> 14, 7m --> 15, 8m --> 16, 9m --> 12, 12m --> 12, 15m --> 12
distance = 6 * 100 * 10 # m --> mm
ir_image_file = './6m/0000_ir.png'
wide_image_file = './6m/0000_wide.png'
zoom_image_file = './6m/0000_zoom.png'
ir_json_file = './6m/0000_ir_6_14.json'
wide_json_file = './6m/0000_wide_6_14.json'
zoom_json_file = './6m/0000_zoom_6_14.json'
#############################
ir_pixel_values = np.array(extract_pixel_values(ir_json_file, ir_image_file))
wide_pixel_values = np.array(extract_pixel_values(wide_json_file, wide_image_file))
zoom_pixel_values = np.array(extract_pixel_values(zoom_json_file, zoom_image_file))
row_amount = ir_pixel_values.shape[0]

print("[INFO] :: Infrared pixel values:\n", ir_pixel_values)
print("[INFO] :: Wide pixel values:\n", wide_pixel_values)
print("[INFO] :: Zoom pixel values:\n", zoom_pixel_values)

depth = math.sqrt(distance**2 - 1680**2)
print(f'[INFO] :: Acquired the depth: {depth} * {row_amount}')
#############################
# intrinsic parameters
K_ir = np.array([
        [1044.03628677823, 0, 335.125645561794],
        [0, 1051.80215540345, 341.579677246452],
        [0, 0, 1]
    ])
K_vi_wide = np.array([
        [2901.19910315714, 0, 940.239619965275],
        [0, 2893.75517626367, 618.475768281058],
        [0, 0, 1]
    ])
#############################
# datas involing the calculation
ir_points = ir_pixel_values
vi_points = wide_pixel_values
K_ir = K_ir
K_vi = K_vi_wide

ir_image = cv2.imread(ir_image_file)
rgb_image = cv2.imread(wide_image_file)
ones_column = np.ones((row_amount, 1), dtype = int)

q_ir_set = np.column_stack((ir_points, ones_column))
q_vi_set = np.column_stack((vi_points, ones_column))
depth_set = np.full((row_amount, 1), depth)
#############################
# Function to print data and type
def print_data_info(data):
    print("Data:", data)
    
    # Check if the data is a NumPy array
    if isinstance(data, np.ndarray):
        print("Type: NumPy Array")
        print("Shape:", data.shape)
        print("Data Type:", data.dtype)
    
    # Check if the data is a Python list
    elif isinstance(data, list):
        print("Type: Python List")
        print("Length:", len(data))
        print("Inner Length:", [len(item) for item in data] if all(isinstance(item, list) for item in data) else "Not a list of lists")
    
    # Check if the data is a tensor (assuming you have a tensor library like PyTorch or TensorFlow)
    elif 'torch' in str(type(data)):  # Example for PyTorch tensors
        print("Type: PyTorch Tensor")
        print("Shape:", data.shape)
        print("Data Type:", data.dtype)
    elif 'tensorflow' in str(type(data)):  # Example for TensorFlow tensors
        print("Type: TensorFlow Tensor")
        print("Shape:", data.shape)
        print("Data Type:", data.dtype)
    
    # If the data type is unknown
    else:
        print("Type: Unknown")

# Test the function with a NumPy array
print_data_info(ir_points)

print("\n")
######################################
# initialized parameters of R, t, (Or R' t')

params_initial = np.random.randn(12)
def residuals(params, vi_points, ir_points, depth):
    R = np.array(params[:9]).reshape(3, 3)
    t = np.array(params[9:])

    # R_prime = np.array(params[:9]).reshape(3, 3)
    # t_prime = np.array(params[9:])

    errors = []
    for q_vi, q_ir, d in zip(q_vi_set, q_ir_set, depth_set):
        # print("q_vi", q_vi)
        # print("q_ir_real", q_ir)
        q_ir_estimated = K_ir @ (R @ np.linalg.inv(K_vi) @ q_vi + (1 / d) * t)
        # print("q_ir_estimated", q_ir_estimated)
        # q_ir_estimated = R_prime @ q_vi + (1 / d) * t_prime
        # q_ir_actual = np.array([uir, vir, 1])

        errors.extend(q_ir_estimated - q_ir)
        # errors += np.linalg.norm(e)
    print("Current residual:", errors)

    return np.array(errors)

# 使用最小二乘法进行优化
result = least_squares(residuals, params_initial, args=(vi_points, ir_points, depth))

# 获取优化后的 R 和 t 参数
optimized_params = result.x
R_opt = optimized_params[:9].reshape(3, 3)
t_opt = optimized_params[9:]

print("Optimized R:", R_opt)
print("Optimized t:", t_opt)
##############################################
# to patch the infrared image on the visible image
R = R_opt
t = t_opt
R_inv = np.linalg.inv(R_opt)
K_ir_inv = np.linalg.inv(K_ir)
d = float(input("[Query] :: Enter the scale factor d:\n "))

R_prime = np.array  ([[0.7194,    0.0048, -370.2559],
                     [-0.0183,    0.7371, -147.8247],
                     [ 0,         0,    1.0000]])

t_prime = np.array  ([6.1169, 18.9934, 0])
# Calculate the inverse matrix
R_prime_inv = np.linalg.inv(R_prime)

######################
# # Calculate the mapping matrix
# M = R_prime_inv
# # Handle the translation component and integrate it into the mapping matrix
# M[:, 2] -= (1 / d) * t_prime

# 计算映射矩阵
M = K_vi @ R_inv @ K_ir_inv
# 处理平移项，将其叠加到映射矩阵
M[:, 2] -= (1 / d) * K_vi @ R_inv @ t
#####################

# 使用 OpenCV 的 warpPerspective 进行透视变换
warped_ir_image = cv2.warpPerspective(ir_image,
                                      M,
                                      (rgb_image.shape[1], rgb_image.shape[0]))
# 调整透明度参数
alpha = 0.4  # 红外图像的透明度（0.0 - 1.0）
beta = 1 - alpha  # RGB 图像的透明度

# 叠加两个图像
blended_image = cv2.addWeighted(rgb_image, alpha, warped_ir_image, beta, 0)
cv2.imshow("Blended Image", blended_image)
cv2.imshow("Converted infrared image", warped_ir_image)
cv2.imshow("original infrared image", ir_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


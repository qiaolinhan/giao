
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


#############################
###### Adjust params here
# distance, 6m --> 14, 7m --> 15, 8m --> 16, 9m --> 12, 12m --> 12, 15m --> 12
Distance = 15
points_amount = 12

distance = Distance * 100 * 10 # m --> mm
d_str = str(Distance)
pa_str = str(points_amount)
ir_image_file = './checkboard/' + d_str + 'm/0000_ir.png'
wide_image_file = './checkboard/' + d_str + 'm/0000_wide.png'
zoom_image_file = './checkboard/' + d_str + 'm/0000_zoom.png'
ir_json_file = './checkboard/' + d_str + 'm/0000_ir_' + d_str + '_' + pa_str + '.json'
wide_json_file = './checkboard/' + d_str + 'm/0000_wide_' + d_str + '_' + pa_str + '.json'
zoom_json_file = './checkboard/' + d_str + 'm/0000_zoom_' + d_str + '_' + pa_str + '.json'
#############################
def extract_pixel_values(json_file, image_file):
    # load the json file
    with open(json_file) as f:
        data = json.load(f)

    image = np.array(Image.open(image_file))
    print(f"[INFO] :: The image shape is: {image.shape}")
    # image = np.array(cv2.imread(image_file))

    pixel_values = []

    for shape in data['shapes']:
        if shape['shape_type'] == 'point':
            # get the coordinate of the point
            point = shape['points'][0]
            w, h = int(point[0]), int(point[1])
            label = shape['label']

            # get the pixel value at the point
            value = image[h, w]

            # append the label and pixel value
            pixel_values.append([w, h])

    return pixel_values

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
####### Adjust params here
# datas involing the calculation
ir_points = ir_pixel_values
vi_points = wide_pixel_values # adjust wide/zoom
K_ir = K_ir
K_vi = K_vi_wide

ir_image = cv2.imread(ir_image_file)
rgb_image = cv2.imread(wide_image_file) # adjust wide/zoom
ones_column = np.ones((row_amount, 1), dtype = int)

q_ir_set = np.column_stack((ir_points, ones_column))
q_vi_set = np.column_stack((vi_points, ones_column))

d = depth
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
# Define the residuals function
def residuals(params, q_vi_set, q_ir_set, d):
    R11, R12, R13, R21, R22, R23, t1, t2 = params
    # R11, R12, R13, R21, R22, R23, R31, R32, R33, t1, t2, t3 = params
    residual_u = []
    residual_v = []
    residual_list = []

    for q_vi, q_ir in zip(q_vi_set, q_ir_set):
        u_vi, v_vi, _ = q_vi
        u_ir, v_ir, _ = q_ir

        # Compute estimated q_ir
        u_ir_hat = R11 * u_vi + R12 * v_vi + R13 + (1 / d) * t1
        v_ir_hat = R21 * u_vi + R22 * v_vi + R23 + (1 / d) * t2
        # Calculate residuals
        error_u = u_ir_hat - u_ir
        error_v = v_ir_hat - v_ir
        residual_u.append(error_u)
        residual_v.append(error_v)
        # residual_1.append(error_1) 
        residual_list.append(error_u)
        residual_list.append(error_v)
    residual_array = np.array(residual_list)
    # print(f"[INFO] :: The residual now shows as:\n {residual_array}")
    return residual_array 
#
# Original LSQR
# result = least_squares(residuals, initial_params, args=(q_vi_set, q_ir_set, d))
# R11, R12, R13, R21, R22, R23, t1, t2 = result.x

# Iterative refinement function
def iterative_refinement(initial_params, 
                         q_vi_set, q_ir_set, d, 
                         max_iter=20, tol=1e-6):
    params = initial_params
    for i in range(max_iter):
        result = least_squares(residuals,
                               params,
                               args=(q_vi_set, q_ir_set, d),
                               loss='huber')
        new_params = result.x
        print(f"{i+1}th iteration")       
        # Check for convergence
        if np.linalg.norm(new_params - params) < tol:
            print(f"Converged after {i+1} iterations")
            break
        params = new_params
    return params
#
# # Initial guess for the parameters
initial_params = np.zeros(8)
# initial_params = np.zeros(12)
#
# # Perform iterative refinement
optimized_params = iterative_refinement(initial_params, q_vi_set, q_ir_set, d)
# # Extract the optimized parameters
R11, R12, R13, R21, R22, R23, t1, t2 = optimized_params
# R11, R12, R13, R21, R22, R23, R31, R32, R33, t1, t2, t3 = optimized_params
# Calculate transformation matrix using all points


print("Optimized R':\n", f"[[{R11}, {R12}, {R13}],\n [{R21}, {R22}, {R23}],\n [0, 0, 1]]")
print("Optimized t':\n", f"[[{t1}],\n [{t2}],\n 0]")

# print("Optimized R':")
# print(f"[[{R11}, {R12}, {R13}],\n [{R21}, {R22}, {R23}],\n [{R31}, {R32}, {R33}]]")
# print("Optimized t':\n", f"[[{t1}],\n [{t2}],\n [{t3}]]")

##############################################
# to patch the infrared image on the visible image
R_prime = np.array([
        [R11, R12, R13],
        [R21, R22, R23],
        [0, 0, 1]
    ])
t_prime = np.array([
       t1, t2, 0 
    ])

##########################
# # Huajun result
# # Computed from MATLAB, 6m, Ready to use
# R_prime = np.array  ([[0.7194,    0.0048, -370.2559],
#                      [-0.0183,    0.7371, -147.8247],
#                      [ 0,         0,    1.0000]])
#
# t_prime = np.array  ([6.1169, 18.9934, 0])
##########################

# Calculate the inverse matrix
R_prime_inv = np.linalg.inv(R_prime)

# ######################
# Compute the mapping matrix
# M = K_vi @ R_inv @ K_ir_inv
# M[:, 2] -= (1 / d) * K_vi @ R_inv @ t

M = R_prime_inv
# Handle the translation component and integrate it into the mapping matrix
M[:, 2] -= (1 / d) * t_prime

######################################
# # Erfan
# # use cv2 to find homography matrix
# M, _ = cv2.findHomography(np.float32(q_ir_set), np.float32(q_vi_set))
#
# # Invert the transformation matrix
# M_inv = cv2.invert(M)[1]
#
#####################################
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
# cv2.imshow("Converted infrared image", warped_ir_image)
# cv2.imshow("original infrared image", ir_image)
cv2.waitKey(0)
cv2.destroyAllWindows()


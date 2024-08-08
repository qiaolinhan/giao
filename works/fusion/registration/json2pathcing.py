
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
import sys
from sklearn.metrics import r2_score,mean_squared_log_error
#############################
# intrinsic parameters
K_ir = np.array([
        [1044.03628677823, 0, 335.125645561794],
        [0, 1051.80215540345, 341.579677246452],
        [0, 0, 1]
    ])
# K_vi_wide = np.array([
#         [2901.19910315714, 0, 940.239619965275],
#         [0, 2893.75517626367, 618.475768281058],
#         [0, 0, 1]
#     ])
K_vi_wide = np.array([
        [1484.39712549035,	0,	964.013870831680],
        [0,	1478.75546438438,	577.707036666613],
        [0,	0,	1]
    ])
#############################
# functions
def extract_pixel_values(json_file, image_file):
    # load the json file
    with open(json_file) as f:
        data = json.load(f)

    image = np.array(Image.open(image_file))
    # print(f"[INFO] :: The image shape is: {image.shape}")
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

# Define the residuals function
# For R_prime and t_prime
def residuals_without_K(params, q_vi_set, q_ir_set, d):
    R_prime_main = params[:6].reshape(2, 3)
    R_prime_bot = np.array([0, 0, 1])
    R_prime = np.vstack((R_prime_main, R_prime_bot))
    t_prime_main = params[6:].reshape(2, 1)
    t_prime = np.vstack((t_prime_main, 0))
    # R_prime = params[:9].reshape(3, 3)
    # t_prime = params[9:].reshape(3, 1)
    residual_u = []
    residual_v = []
    residual_list = []

    for q_vi, q_ir in zip(q_vi_set, q_ir_set):
        q_vi = q_vi.reshape(3, 1)
        q_ir = q_ir.reshape(3, 1)

        q_ir_hat = R_prime @ q_vi + (1 / d) * t_prime

        # Calculate residuals
        error_u = q_ir_hat[0] - q_ir[0]
        error_v = q_ir_hat[1] - q_ir[1]
        residual_u = error_u.flatten()
        residual_v = error_v.flatten()

        residual_list.append(error_u)
        residual_list.append(error_v)
    residual_array = np.array(residual_list).flatten()
    # print(f"[INFO] :: The residual now shows as:\n {residual_array}")
    return residual_array 

# For R and t
def residuals_with_K(params, q_vi_set, q_ir_set, d):
    R_main = params[:6].reshape(2, 3)
    R_bot = np.array([0, 0, 1])
    R = np.vstack((R_main, R_bot))
    t_main = params[6:].reshape(2, 1)
    t = np.vstack((t_main, 0))
    # R11, R12, R13, R21, R22, R23, R31, R32, R33, t1, t2, t3 = params
    residual_u = []
    residual_v = []
    residual_list = []

    for q_vi, q_ir in zip(q_vi_set, q_ir_set):
        q_vi_h = q_vi.reshape(3, 1)
        q_ir_h = q_ir.reshape(3, 1)

        # Compute estimated q_ir
        q_ir_hat = K_ir @ R @ K_vi_inv @ q_vi + (1 / d) * K_vi @ t        

        # Calculate residuals
        error_u = q_ir_hat[0] - q_ir[0]
        error_v = q_ir_hat[1] - q_ir[1]
        residual_u = error_u.flatten()
        residual_v = error_v.flatten()
        # residual_1.append(error_1) 
        residual_list.append(residual_u)
        residual_list.append(residual_v)
    residual_array = np.array(residual_list).flatten()
    # print(f"[INFO] :: The residual now shows as:\n {residual_array}")
    return residual_array 

# Iterative refinement function
# Original LSQR
# result = least_squares(residuals, initial_params, args=(q_vi_set, q_ir_set, d))
# R11, R12, R13, R21, R22, R23, t1, t2 = result.x
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

# To predict the 3D points coordinates in infrared images
def predict_ir_points(params, q_vi_set, d):
    R_prime_main = params[:6].reshape(2, 3)
    R_prime_bot = np.array([[0, 0, 1]])
    R_prime = np.vstack((R_prime_main, R_prime_bot))
    t_prime_main = params[6:].reshape(2, 1)
    t_prime_bot = np.array([[0]])
    t_prime = np.vstack((t_prime_main, t_prime_bot))

    q_ir_hat_set = []
    uv_ir_hat_set = []
    for q_vi in zip(q_vi_set):
        # transformed 1x3 --> 3x1
        q_vi = np.array(q_vi).T

        q_ir_hat = R_prime @ q_vi + (1 / d) * t_prime
        q_ir_hat_set.append(q_ir_hat) 

    return np.array(q_ir_hat_set)

# To compute the evaluation metrics
def evaluation(params, q_vi_set, q_ir_set, d):
    # estimate the ir point coordinates (homonenous)
    q_ir_hat_set = predict_ir_points(params, q_vi_set, d)
    q_ir_hat_set = np.array(q_ir_hat_set)
    # q_ir_hat_set = q_ir_hat_set.squeeze(axis=-1) 
    # transform 14x2x1--> 14x2
    q_ir_hat_set = q_ir_hat_set.reshape(q_ir_hat_set.shape[0], -1)
    # print_data_info(q_ir_hat_set[:, :2])

    q_ir_set = np.array(q_ir_set)
    # print_data_info(q_ir_set[:, :2])

    # # get the ir point coordinates
    # uv_ir_set = q_ir_set[:2]
    # uv_ir_set_list = [arr.tolist() for arr in uv_ir_set]

    residual_set = q_ir_hat_set[:, :2] - q_ir_set[:, :2]
    print_data_info(residual_set)

    rss = np.sum(residual_set ** 2)
    rmse = np.sqrt(np.mean(residual_set ** 2))
    mae = np.mean(np.abs(residual_set))
    r2 = r2_score(q_ir_set.flatten(), q_ir_hat_set.flatten())
    msle = mean_squared_log_error(q_ir_set.flatten(), q_ir_hat_set.flatten())

    return rss, rmse, mae, r2, msle

# Function to print data and type
def print_data_info(data):
    print("[INFO] :: Information of the data show as follows:\n")
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
#############################
###### Step1: Specify the distance and points amount to use for optimization
while True:
    try:
        Distance = int(input("[QUERY] :: Please specify the dataset distance to use. (6/7/8/9/12/15m):\n"))
        if Distance == 6:
            point_amount = 14
            break
        elif Distance == 7:
            point_amount = 15
            break
        elif Distance == 8:
            point_amount = 16
            break
        elif Distance == 9 or Distance == 12 or Distance == 15:
            point_amount = 12
            break
        else:
            print("[ERROR] :: Invalid distance. (6/7/8/9/12/15)")
    except ValueError:
        print("[INFO] :: Please enter a valid number.")

distance = int(Distance) * 100 * 100 # m --> mm
d_str = str(Distance)
pa_str = str(point_amount)
ir_image_file = './checkboard/' + d_str + 'm/0000_ir.png'
wide_image_file = './checkboard/' + d_str + 'm/0000_wide.png'
zoom_image_file = './checkboard/' + d_str + 'm/0000_zoom.png'
ir_json_file = './checkboard/' + d_str + 'm/0000_ir_' + d_str + '_' + pa_str + '.json'
wide_json_file = './checkboard/' + d_str + 'm/0000_wide_' + d_str + '_' + pa_str + '.json'
zoom_json_file = './checkboard/' + d_str + 'm/0000_zoom_' + d_str + '_' + pa_str + '.json'

ir_pixel_values = np.array(extract_pixel_values(ir_json_file, ir_image_file))
wide_pixel_values = np.array(extract_pixel_values(wide_json_file, wide_image_file))
zoom_pixel_values = np.array(extract_pixel_values(zoom_json_file, zoom_image_file))
row_amount = ir_pixel_values.shape[0]

# print("[INFO] :: Infrared pixel values:\n", ir_pixel_values)
# print("[INFO] :: Wide pixel values:\n", wide_pixel_values)
# print("[INFO] :: Zoom pixel values:\n", zoom_pixel_values)

depth = math.sqrt(distance**2 - 1680**2)
print(f"[INFO] :: Selected dataset at distance: {Distance}m, 3D points amount: {point_amount}.")
print(f'[INFO] :: Acquired the depth: {depth}')

###### Step2: Specify infrared images warp on wide images or zoom images
while True:
    try:
        Patch = str(input("[QUERY] :: Please specify, want to warp infrared images on wide or zoom images. (Wide/Zoom):\n")).strip().lower()
        if Patch in ["wide", "w"]:
            print("[INFO] :: Computing infrared-wide matrix")
            # datas involing the calculation
            ir_points = ir_pixel_values
            vi_points = wide_pixel_values # adjust wide/zoom
            K_ir = K_ir
            K_vi = K_vi_wide
            ir_image = cv2.imread(ir_image_file)
            rgb_image = cv2.imread(wide_image_file) # adjust wide/zoom
            break
        elif Patch in ["zoom", "z"]:
            print("[INFO] :: Computing infrared-zoom matrix")
            # datas involing the calculation
            ir_points = ir_pixel_values
            vi_points = zoom_pixel_values # adjust wide/zoom
            K_ir = K_ir
            K_vi = K_vi_zoom
            ir_image = cv2.imread(ir_image_file)
            rgb_image = cv2.imread(zoom_image_file) # adjust wide/zoom
            break
        else:
            print("[ERROR] :: Invalid input, try again. (Wide/Zoom, W/Z)?")
    except ValueError:
        print("[INFO] :: Please enter a valid camera type.")

ones_column = np.ones((row_amount, 1), dtype = int)


d = depth
depth_set = np.full((row_amount, 1), depth)
q_ir_set = np.column_stack((ir_points, ones_column))
q_vi_set = np.column_stack((vi_points, ones_column))

uv_ir_set = q_ir_set[:2]
# uv_ir_hat_set = q_ir_hat_set[:2]

print("[INFO] :: uv_ir_set")
print_data_info(uv_ir_set)

K_ir_inv = np.linalg.inv(K_ir)
K_vi_inv = np.linalg.inv(K_vi)
#############################

# Test the function with a NumPy array
print_data_info(ir_points)

###### Step3: Specify avioding intrinsic parameters K_vi and K_ir or not
while True:
    try:
        Avoi = str(input("[QUERY] :: Please specify, aviod using intrincics or not? (Yes/No):\n")).strip().lower()

        if Avoi in ["yes", "y"]:
            residuals = residuals_without_K
            print("[INFO] :: Estimating R_prime and t_prime")
            initial_params = np.zeros(8)
            # # Perform iterative refinement
            optimized_params = iterative_refinement(initial_params, q_vi_set, q_ir_set, d)
            # # Extract the optimized parameters
            R11, R12, R13, R21, R22, R23, t1, t2 = optimized_params
            # R11, R12, R13, R21, R22, R23, R31, R32, R33, t1, t2, t3 = optimized_params
            # to patch the infrared image on the visible image
            R_prime = np.array([
                    [R11, R12, R13],
                    [R21, R22, R23],
                    [0.000, 0.000, 1.000]
                ])
            t_prime = np.array([
                   t1, t2, 0.000 
                ])
            print("Optimized R':\n", f"[[{R11}, {R12}, {R13}],\n [{R21}, {R22}, {R23}],\n [0.000, 0.000, 1.000]]")
            print("Optimized t':\n", f"[[{t1}],\n [{t2}],\n [0.000]]")
             ##########################
            # # Huajun result
            # # Computed from MATLAB, 6m, Ready to use
            # R_prime = np.array  ([[0.7194,    0.0048, -370.2559],
            #                      [-0.0183,    0.7371, -147.8247],
            #                      [ 0,         0,    1.0000]])
            #
            # t_prime = np.array  ([6.1169, 18.9934, 0])
           
            # params_eval = []
            # R_prime_flatten = R_prime.flatten()
            # params_eval.append(R_prime_flatten)
            # t_prime_flatten = t_prime.flatten()
            # params_eval.append(t_prime_flatten)
            
            ##########################
            # evaluations
            rss, rmse, mae, r2, msle = evaluation(optimized_params, q_vi_set, q_ir_set, d)

            print(f"RSS: {rss}")
            print(f"RMSE: {rmse}")
            print(f"MAE: {mae}")
            print(f"R2: {r2}")
            print(f"MSLE: {msle}")
            
            #########################
            # Patching result part
            alpha = float(input("[QUERY] :: Please specify alpha value for infrared image patching. (0 < alpha < 1):\n"))

            # Calculate the inverse matrix
            R_prime_inv = np.linalg.inv(R_prime)
            M = R_prime_inv
            # Handle the translation component and integrate it into the mapping matrix
            M[:, 2] -= (1 / d) * t_prime
            warped_ir_image = cv2.warpPerspective(ir_image,
                                                  M,
                                                  (rgb_image.shape[1], rgb_image.shape[0]))

            blended_image = cv2.addWeighted(rgb_image, alpha, warped_ir_image, 1 - alpha, 0)
            cv2.imshow("Blended Image", blended_image)
            # cv2.imshow("Converted infrared image", warped_ir_image)
            # cv2.imshow("original infrared image", ir_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break

        # The K based scheme does not work well
        elif Avoi in ["no", "n"]:
            print("[INFO] :: Estimating R and t while using K_vi and K_ir")
            residuals = residuals_with_K
            initial_params = np.zeros(8)
            # # Perform iterative refinement
            optimized_params = iterative_refinement(initial_params, q_vi_set, q_ir_set, d)
            # # Extract the optimized parameters
            R11, R12, R13, R21, R22, R23, t1, t2 = optimized_params
            R = np.array([
                    [R11, R12, R13],
                    [R21, R22, R23],
                    [0.00, 0.00, 1.00]
                ])
            t = np.array([
                    t1, t2, 0.00
                ])
            print("Optimized R:\n", f"[[{R11}, {R12}, {R13}],\n [{R21}, {R22}, {R23}],\n [0.00, 0.00, 1.00]]\n")
            print("Optimized t:\n", f"[[{t1}],\n [{t2}],\n [0.00]]\n")
            
            alpha = float(input("[QUERY] :: Please specify alpha value for infrared image patching. (0 < alpha < 1):\n"))
            ######################
            R_inv = np.linalg.inv(R)
            # Compute the mapping matrix
            M = K_vi @ R_inv @ K_ir_inv
            M[:, 2] -= (1 / d) * K_vi @ t
            warped_ir_image = cv2.warpPerspective(ir_image,
                                                  M,
                                                  (rgb_image.shape[1], rgb_image.shape[0]))
            
            blended_image = cv2.addWeighted(rgb_image, alpha, warped_ir_image, 1 - alpha, 0)
            cv2.imshow("Blended Image", blended_image)
            # cv2.imshow("Converted infrared image", warped_ir_image)
            # cv2.imshow("original infrared image", ir_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            break
        else:
            print("[ERROR] :: Invalid specification, use intrinsics or not? (Yes/No)")
    except ValueError as e:
        print(f"[INFO] :: {e}, Avoid K_vi, K_ir or not. (Yes/No)")

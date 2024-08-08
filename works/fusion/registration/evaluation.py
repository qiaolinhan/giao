# This is to evaluate the performance of the registration results and the estimited
# homography matrix

import numpy as np
import cv2
from sklearn.metrics import r2_score, mean_squared_log_error
# from scipy.optimize import least_square

######
# step0: load necessary functions

# To compute the depth with the loaded distance
def dis2depth(distance):
    distance = distance * 10000
    depth = np.sqrt(distance**2 - 16800**2)
    return depth
# To predict the 3D points coordinates in infrared images
def predict_ir_points(params, q_vi_set, distances):
    R = params[:9].reshape(3, 3)
    t = params[9:].reshape(3, 1)
    q_ir_hat_set = []

    for q_vi, d in zip(q_vi_set, depth):
        q_ir_hat = R_prime @ q_vi + (1 / d) * t_prime
        q_ir_hat_set.append(q_ir_hat.flatten()) 

    return np.array(q_ir_hat_set)

# To compute the evaluation metrics
def evaluatoin_transformation(params, q_vi_set, q_ir_set, depth):
    q_ir_hat_set = predict_ir_points(params, q_vi_set, depth)
    q_ir_set = np.array(q_ir_set)

    residual_set = q_ir_set - q_ir_hat_set

    rss = np.sum(residuals ** 2)
    rmse = np.sqrt(np.mean(residual_set))
    mae = np.mean(np.abs(residual_set))
    r2 = r2_score(q_ir_set.flatten(), q_ir_hat_set.flatten())
    msle = mean_squared_log_error(q_ir_set.flatten(), q_ir_hat_set.flatten())

    return rss, rmse, mae, r2, msle
######
# step1: load the depth, 3D point coordinates, estimated params of R' and t'
while True:
    try:
        Distance = int(input("[QUERY] :: Please input the distance to load estimated params and original 3D point coordinates. (6/7/8/9/12/15):\n"))

        if Distance == 6:
            d = dis2depth(Distance)
            optimized_params_w = 
            optimized_params_z = 
            break
        elif Distance == 7:
            d = dis2depth(Distance)
            optimized_params_w = 
            optimized_params_z = 
            break
        elif Distance == 8:
            d = dis2depth(Distance)
            optimized_params_w = 
            optimized_params_z = 
            break
        elif Distance == 9:
            d = dis2depth(Distance)
            optimized_params_w = 
            optimized_params_z = 
            break
        elif Distance == 12:
            d = dis2depth(Distance)
            optimized_params_w = 
            optimized_params_z = 
            break
        elif Distance == 15:
            d = dis2depth(Distance)
            optimized_params_w = 
            optimized_params_z = 
            break
        else:
            print("[Error] :: Invalid input.")
    except ValueError:
        print("[QUERY] :: Try again the distance input. (6/7/8/9/12/15)")

# Based on the acquired R' and t' to convert infrared image and patch on wide/zoom
# images

import cv2
import numpy as np
import cv2
import math

#############################
##### Params to adjust
alpha = 0.7
ir_video_path = "./set3/ir_20240604.MP4"
wide_video_path = "./set3/wide_20240604.MP4"
zoom_video_path = "./set3/zoom_20240604.MP4"

output_path = '15m_blend_wide.mp4'

cap_ir = cv2.VideoCapture(ir_video_path)
cap_vi = cv2.VideoCapture(wide_video_path) # wide/zoom
# adjust distance to have appropriate depth
# 6, 7, 8, 9, 12, 15
# distance = 6 * 1000
# R_prime = np.array([[0.7194,    0.0048, -370.2559],
#                      [-0.0183,    0.7371, -147.8247],
#                      [ 0,         0,    1.0000]])
#
# t_prime = np.array([6.1169, 18.9934, 0])

distance = 15 * 1000
# # zoom <---> ir
# R_prime = np.array([
#     [0.39736810838786757, 0.004310655569585121, -47.79727809898252],
#     [-0.00627911525394201, 0.3786220554710437, 33.22929031768907],
#     [0, 0, 1]
#     ])
# t_prime = np.array ([
#     -168.48549151995718, -5.945557749661148, 0
#     ])
# wide <---> ir
R_prime = np.array([
    [0.7479685738237563, 0.013424881602741103, -404.9740191363077],
    [-0.013472686218325609, 0.7209636740568596, -141.2487787228108],
    [0, 0, 1]
    ])
t_prime = np.array([
    -269.1662934724526, 96.19960945255941, 0
    ])
#############################
d = math.sqrt(distance**2 - 1680**2)
# Calculate the inverse matrix
R_prime_inv = np.linalg.inv(R_prime)
M = R_prime_inv
# Handle the translation component and integrate it into the mapping matrix
M[:, 2] -= (1 / d) * t_prime
############################
# Get the width, height, and frames per second (fps) of the video
frame_width = int(cap_vi.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap_vi.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap_vi.get(cv2.CAP_PROP_FPS)

# Define the codec and create a VideoWriter object to save the processed video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID', 'DIVX', 'X264', etc.
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
#############################
# To blend
def blend(frame1, frame2, alpha = 0.5):
    return cv2.addWeighted(frame1, alpha, frame2, 1 - alpha, 0)

while cap_ir.isOpened() and cap_vi.isOpened():
    ret_ir, frame_ir = cap_ir.read()
    ret_vi, frame_vi = cap_vi.read()

    if not ret_ir or not ret_vi or frame_ir is None or frame_vi is None:
        print("[ERROR] :: Videos are not loaded correctly")
        break
    

    frame_ir_warp = cv2.warpPerspective(frame_ir,
                                          M,
                                          (frame_vi.shape[1], frame_vi.shape[0]))

    combined_frame = blend(frame_ir_warp, frame_vi, alpha)

    cv2.imshow('Video after registration', combined_frame)

    # Write the combined frame to the output video
    out.write(combined_frame)

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap1.release()
cap2.release()
out.release()
cv2.destroyAllWindows()

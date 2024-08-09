# Based on the acquired R' and t' to convert infrared image and patch on wide/zoom
# images

import cv2
import numpy as np
import cv2
import math

#############################
##### Params to adjust
alpha = 0.5
ir_video_path = "./set3/ir_20240604.MP4"
wide_video_path = "./set3/wide_20240604.MP4"
zoom_video_path = "./set3/zoom_20240604.MP4"

# ir_video_path = "./set3/15_T.MP4"
# wide_video_path = "./set3/15_W.MP4"

output_path = '12m_blend_wide.mp4'

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

distance = 12
# # zoom <---> ir, 12m
# R_prime = np.array([
#     [0.38711162594795556, 0.004771407855517256, -36.81910331340785],
#     [-0.007726461324734653, 0.3892775375724707, 29.57225072650655],
#     [0.000, 0.000, 1.000]
#     ])
# t_prime = np.array([
#     1.9597560410526582,
#     218.59099225699518,
#     0.000
#     ])
########################
# [IMPORTANT] For indoor
# wide <---> ir
# R_prime = np.array([
#    [0.7393831824732047, 0.007377260736396099, -393.8937141863329],
#  [-0.017384336114754814, 0.7475328230776862, -155.40287561641583],
#  [0.000, 0.000, 1.000] 
#     ])
# t_prime = np.array([
#  15.863211885625805,
#  43.15765165417564,
#  0.000  ])
##########################
# Optimized R':
#  [[0.7393831824732047, 0.007377260736396099, -393.8937141863329],
#  [-0.017384336114754814, 0.7475328230776862, -155.40287561641583],
#  [0.000, 0.000, 1.000]]
# Optimized t':
#  [[15.863211885625805],
#  [43.15765165417564],
#  [0.000]]
##############################
# [IMPORTANT] For outdoor
R_prime = np.array([
    [0.7393831824732047, 0.007377260736396099, -393.8937141863329],
    [-0.017384336114754814, 0.7475328230776862, -155.40287561641583],
    [0.000, 0.000, 1.000]
    ])
t_prime = np.array([
    295.863211885625805,
    268.15765165417564,
    0.000   ])
#############################
d = math.sqrt(distance**2 - 1.68**2)
# Calculate the inverse matrix
R_prime_inv = np.linalg.inv(R_prime)
M = R_prime_inv
# Handle the translation component and integrate it into the mapping matrix
M[:, 2] -= (1 / d) * t_prime
print("M:\n",M)
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
cap_vi.release()
cap_ir.release()
out.release()
cv2.destroyAllWindows()

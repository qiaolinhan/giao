# this is a image fusion program based on 
# OpenCv original packages
# Linhan Qiao
# 20240201
import os
import cv2
import numpy as np

Path = "/home/qiao/dev/datasets/original_datasets/FlightClub20220528"

vi_img_path = Path + "/m300_grabbed_data_1_17.1/rgb/0.png"
ir_img_path = Path + "/m300_grabbed_data_1_17.1/ir/0.png"

print("[INFO]: vi_path:", vi_img_path)
# Load visible image and ir image
vi_original = cv2.imread(vi_img_path)
ir_original = cv2.imread(ir_img_path)


if vi_original is None or ir_original is None:
    print("[ERROR]: Loading image fail!")
else:
    print("[INFO]: Loading images succeed!")
# converting IR image into grayscaled one
ir_gray = ir_original
# ir_gray = cv2.cvtColor(ir_original, cv2.COLOR_BGR2GRAY)

image1 = vi_original
image2 = ir_gray

# blending fusion: img1, alpha, img2
alpha = 0.5
fusion = cv2.addWeighted(image1, alpha, image2, 1 - alpha, 0)

# display the original images and the processed image
cv2.imshow("vi_original", vi_original)
cv2.imshow("ir_original", ir_original)
cv2.imshow("ir_grayscale", ir_gray)
cv2.imshow("VI fusion", fusion)

cv2.waitKey(0)
cv2.destroyAllWindows()

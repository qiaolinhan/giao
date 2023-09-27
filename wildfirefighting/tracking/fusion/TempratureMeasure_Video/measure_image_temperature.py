# qiaolinhan 20230920
import numpy as np 
import cv2 

# open the gray16 image
gray16_image = cv2.imread('sample_ir.png', cv2.IMREAD_ANYDEPTH) 

# define the pixel coordinates 
x = 730 
y = 320

# # create mouse global coordinates 
# x_mouse = 0 
# y_mouse = 0 
#
# # create thermal video fps variable (8 fps in this case) 
# fps = 8
#
# # mouse event function 
# def mouse_events(event, x, y, flags, param):
#     # mouse movement event 
#     if event == cv2.EVENT_MOUSEMOVE:
#
#         # update mouse global coordinates 
#         global x_mouse 
#         global y_mouse 
#
#         x_mouse = x 
#         y_mouse = y
#
# to calculate the temperature
# TODO: a thermal camera is needed 
#
# in Kelvin
pixel_flame_gray16_0 = gray16_image[y, x] 
# in Celsius
pixel_flame_gray16_c = (pixel_flame_gray16_0 / 100) - 273.15 
# in Fahrenheit
pixel_flame_gray16_f = (pixel_flame_gray16_0 / 100) * 9 / 5 - 459.67 

# Convert the gray16 image into gray8 to show the result 
gray8_image = np.zeros((120, 160), dtype = np.uint8)
gray8_image = cv2.normalize(gray16_image, gray8_image, 0, 255, cv2.NORM_MINMAX) 
gray8_image = np.uint8(gray8_image)

# write a pointer in the image 
cv2.circle(gray8_image, (x, y), 2, (0, 0 , 0), -1)
cv2.circle(gray16_image, (x, y), 2, (0, 0, 0), -1)

# write tempreture value in gray8 and gray16 image
font = cv2.FONT_HERSHEY_SIMPLEX
fontscale = 1
color = (0, 0, 255)
thickness = 2
cv2.putText(gray8_image, "{0:.1f} Celsius".format(pixel_flame_gray16_c), (x - 40, y - 10), font, fontscale, color, thickness, cv2.LINE_AA)
cv2.putText(gray16_image, "{0:.1f} Fahrenheit".format(pixel_flame_gray16_f), (x - 80, y - 15),font, fontscale, color, thickness, cv2.LINE_AA)

# show result 
cv2.imshow("gray8-calsius", gray8_image)
cv2.imshow("gray16-fahrenheit", gray16_image)
cv2.waitKey(0)


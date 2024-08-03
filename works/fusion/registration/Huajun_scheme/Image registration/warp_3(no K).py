import numpy as np
import cv2

# Load infrared and visible light images
ir_image = cv2.imread('ir_test2.png')
rgb_image = cv2.imread('vs_test2.png')

R_prime = np.array  ([[0.7194,    0.0048, -370.2559],
                     [-0.0183,    0.7371, -147.8247],
                     [ 0,         0,    1.0000]])

t_prime = np.array  ([6.1169, 18.9934, 0])

# Input scale factor d
d = float(input("Enter the scale factor d: "))

# Calculate the inverse matrix
R_prime_inv = np.linalg.inv(R_prime)

# Calculate the mapping matrix
M = R_prime_inv

# Handle the translation component and integrate it into the mapping matrix
M[:, 2] -= (1 / d) * t_prime

# Use OpenCV's warpPerspective to perform perspective transformation
warped_ir_image = cv2.warpPerspective(ir_image, M, (rgb_image.shape[1], rgb_image.shape[0]))

# Adjust the transparency parameter
alpha = 0.4  # Transparency of the infrared image (0.0 - 1.0)
beta = 1 - alpha  # Transparency of the RGB image

# Blend the two images
blended_image = cv2.addWeighted(rgb_image, alpha, warped_ir_image, beta, 0)

# Display the results
# cv2.imshow("Warped IR Image", warped_ir_image)
# cv2.imshow("RGB Image", rgb_image)
cv2.imshow("Blended Image", blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

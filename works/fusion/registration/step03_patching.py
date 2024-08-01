# Based on the acquired R' and t' to convert infrared image and patch on wide/zoom
# images

import cv2
import numpy as np

R11, R12, R13, R21, R22, R23, t1, t2 = parames[:8]

transformation_matrix = np.array([
        [R11, R12, t1],
        [R21, R22, t2],
        [0, 0, 1]
    ])

# translation_transformation_matrix = np.array([
#         [t1],
#         [t2],
#         [0]
#     ])

# load the images
ir_image_ir_path = "./12m/0000_ir.png"
vi_image_wide_path = "./12m/0000_wide.png"
vi_image_zoom_path = "./12m/0000_zoom.png"

ir_image_ir = cv2.imread(ir_image_path)
vi_image_wide = cv2.imread(vi_image_wide_path)
vi_image_zoom = cv2.imread(vi_image_zoom)

# apply the trasnformation to the infrared image
height_w, width_w, _ = vi_image_wide.shape
height_z, width_z, _ = vi_image_zoom.shape
height_ir, width_ir, _ = ir_image_ir.shape

transformed_ir_image = cv2.warpPerspective(ir_image_ir, transforamtion_matrix,
                                           (width_ir, height_ir))

# create a mask for the transformed infrared image
ir_mask = np.any(transformed_ir_image > 0, axis=-1).astype(np.uint8) * 255

# Find contours of the mask to get the bounding box
contours, _ = cv2.findContours(ir_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
x, y, w, h = cv2.boundingRect(contours[0])

# Crop the transformed infrared image to the bounding box
cropped_transformed_ir_image = transformed_ir_image[y:y+h, x:x+w]

# Get the size of the cropped transformed infrared image
height_cropped_ir, width_cropped_ir, _ = cropped_transformed_ir_image.shape

# Define the top-left corner where the infrared image will be placed on the visible image
top_left_x = (vi_image_wide.shape[1] - width_cropped_ir) // 2
top_left_y = (vi_image_wide.shape[0] - height_cropped_ir) // 2

# Overlay the cropped transformed infrared image onto the visible image
result_image = visible_image.copy()
result_image[top_left_y:top_left_y+height_cropped_ir, top_left_x:top_left_x+width_cropped_ir] = cropped_transformed_ir_image

# Display the result
plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
plt.title('Overlay Image')
plt.show()

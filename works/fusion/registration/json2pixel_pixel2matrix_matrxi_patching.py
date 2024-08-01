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
distance = 12 * 100 * 10 # m --> mm
ir_image_file = './6m/0000_ir.png'
wide_image_file = './6m/0000_wide.png'
zoom_image_file = './6m/0000_zoom.png'
ir_json_file = './6m/0000_ir_6_14.json'
wide_json_file = './6m/0000_wide_6_14.json'
zoom_json_file = './6m/0000_zoom_6_14.json'
#############################

depth = math.sqrt(distance**2 - 1680**2)
print(f'[INFO] :: Acquired the depth: {depth}')

ir_pixel_values = extract_pixel_values(ir_json_file, ir_image_file)
wide_pixel_values = extract_pixel_values(wide_json_file, wide_image_file)
zoom_pixel_values = extract_pixel_values(zoom_json_file, zoom_image_file)

print("[INFO] :: Infrared pixel values:\n", ir_pixel_values)
print("[INFO] :: Wide pixel values:\n", wide_pixel_values)
print("[INFO] :: Zoom pixel values:\n", zoom_pixel_values)

vi_points_wide = wide_pixel_values
vi_points_zoom = zoom_pixel_values
ir_points_ir = ir_pixel_values
d = depth

definition = input("[Query] :: Do you need to set R31, R32, R33, t3 as 0, 0, 1, 0?(Y/N)\n")

# If defining R31=0, R32 = 0, R33 = 1, t3 =0
if definition == 'Y':
#######################
    # construct R' and t'
    R_w = []
    t_w = []

    # wide and infrared
    for (x1, y1), (x2, y2) in zip(ir_points_ir, vi_points_wide):
        R_w.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        R_w.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])
        # R_w.append([0, 0, 0, 0, 0, 0, x1, y1, 1, -1*x1, -1*y1, -1])
        # Removing the row associated with z coordinate transformation
        # and setting R33 = 1, R31 = 0, R32 = 0, t3 = 0
        # This results in R31, R32, t3 not appearing in the linear system

        t_w.extend([x2, y2])

    R_w = np.array(R_w)
    t_w = np.array(t_w)

    # solve for the parameters using least squares
    params, _, _, _ = np.linalg.lstsq(R_w, t_w, rcond=None)

    # extract the transformation matrix components
    # Manually set R31=R32=0, R33=1
    R11, R12, R13, R21, R22, R23, R31, R32, R33, t1, t2, t3 = np.append(params, [0, 0, 1])  

    print("[INFO] :: For infrared images to wide images")
    print("[INFO] :: Transform matrix:\n", np.array([[R11, R12, R13], [R21, R22, R23],
                                                     [0, 0, 1]]))
    print("[INFO] :: Translation vector (scaled with d):", np.array([t1, t2, 0]))

    transformation_matrix_ir2w = np.array([
            [R11, R12, t1],
            [R21, R22, t2],
            [0, 0, 1]
        ])

    # load the images and apply the trasnformation to the infrared image
    ir_image_ir = cv2.imread(ir_image_file)
    height_ir, width_ir, _ = ir_image_ir.shape

    vi_image_wide = cv2.imread(wide_image_file)
    height_w, width_w, _ = vi_image_wide.shape


    transformed_ir_image = cv2.warpPerspective(ir_image_ir,
                                               transformation_matrix_ir2w,
                                               (width_ir, height_ir))
    # Check if the transformed infrared image is valid
    if np.all(transformed_ir_image == 0):
        print("The transformation matrix might be incorrect.")

    # Display the original infrared image
    plt.figure()
    plt.imshow(cv2.cvtColor(ir_image_ir, cv2.COLOR_BGR2RGB))
    plt.title('Original infrared image')
    plt.show()

    # Display the transformed infrared image
    plt.figure()
    plt.imshow(cv2.cvtColor(transformed_ir_image, cv2.COLOR_BGR2RGB))
    plt.title('Transformed infrared image')
    plt.show()

    # Adjust the transparency parameter
    alpha = 0.4  # Transparency of the infrared image (0.0 - 1.0)
    beta = 1 - alpha  # Transparency of the RGB image

    # Blend the two images
    blended_image = cv2.addWeighted(vi_image_wide, alpha, transformed_ir_image, beta, 0)

    # Display the results
    # cv2.imshow("Warped IR Image", warped_ir_image)
    # cv2.imshow("RGB Image", rgb_image)
    cv2.imshow("Blended Image", blended_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

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
    result_image = vi_image_wide.copy()
    result_image[top_left_y:top_left_y+height_cropped_ir, top_left_x:top_left_x+width_cropped_ir] = cropped_transformed_ir_image

    # Display the result
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Overlay infrared image on wide image')
    plt.show()

    ##########################
    # construct R' and t'
    R_z = []
    t_z = []

    # wide and infrared
    for (x1, y1), (x2, y2) in zip(ir_points_ir, vi_points_zoom):
        R_z.append([x1, y1, 1, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        R_z.append([0, 0, 0, x1, y1, 1, -y2*x1, -y2*y1, -y2])
        # R_z.append([0, 0, 0, 0, 0, 0, x1, y1, 1, -1*x1, -1*y1, -1])
        # Removing the row associated with z coordinate transformation
        # and setting R33 = 1, R31 = 0, R32 = 0, t3 = 0
        # This results in R31, R32, t3 not appearing in the linear system

        t_z.extend([x2, y2])

    R_z = np.array(R_z)
    t_z = np.array(t_z)

    # solve for the parameters using least squares
    params, _, _, _ = np.linalg.lstsq(R_z, t_z, rcond=None)

    # extract the transformation matrix components
    # Manually set R31=R32=0, R33=1
    R11, R12, R13, R21, R22, R23, R31, R32, R33, t1, t2, t3 = np.append(params, [0, 0, 1])  
      

    print("[INFO] :: For infrared images to zoom images")
    print("[INFO] :: Transform matrix:\n", np.array([[R11, R12, R13], [R21, R22, R23],
                                                     [0, 0, 1]]))
    print("[INFO] :: Translation vector (scaled with d):", np.array([t1, t2, 0]))

    transformation_matrix_ir2z = np.array([
            [R11, R12, t1],
            [R21, R22, t2],
            [0, 0, 1]
        ])

    ir_image_ir = cv2.imread(ir_image_file)
    height_ir, width_ir, _ = ir_image_ir.shape

    vi_image_zoom = cv2.imread(zoom_image_file)
    height_z, width_z, _ = vi_image_zoom.shape

    transformed_ir_image = cv2.warpPerspective(ir_image_ir,
                                               transformation_matrix_ir2z,
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
    top_left_x = (vi_image_zoom.shape[1] - width_cropped_ir) // 2
    top_left_y = (vi_image_zoom.shape[0] - height_cropped_ir) // 2

    # Overlay the cropped transformed infrared image onto the visible image
    result_image = vi_image_zoom.copy()
    result_image[top_left_y:top_left_y+height_cropped_ir, top_left_x:top_left_x+width_cropped_ir] = cropped_transformed_ir_image

    # Display the result
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Overlay infrared image on zoom image')
    plt.show()

# If not setting R31 = 0, R32 = 0, R33 = 1, t3 = 0 
else:
#######################
    # construct R' and t'
    R_w = []
    t_w = []

    # wide and infrared
    for (x1, y1), (x2, y2) in zip(ir_points_ir, vi_points_wide):
        R_w.append([x1, y1, 1, 0, 0, 0, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        R_w.append([0, 0, 0, x1, y1, 1, 0, 0, 0, -y2*x1, -y2*y1, -y2])
        R_w.append([0, 0, 0, 0, 0, 0, x1, y1, 1, -1*x1, -1*y1, -1])
        # Removing the row associated with z coordinate transformation
        # and setting R33 = 1, R31 = 0, R32 = 0, t3 = 0
        # This results in R31, R32, t3 not appearing in the linear system

        t_w.extend([x2-1/d, y2-1/d, 1-1/d])

    R_w = np.array(R_w)
    t_w = np.array(t_w)

    # solve for the parameters using least squares
    params, _, _, _ = np.linalg.lstsq(R_w, t_w, rcond=None)

    # extract the transformation matrix components
    # Manually set R31=R32=0, R33=1
    R11, R12, R13, R21, R22, R23, R31, R32, R33, t1, t2, t3 = params  

    print("[INFO] :: For infrared images to wide images")
    print("[INFO] :: Transform matrix:\n", np.array([[R11, R12, R13], [R21, R22, R23],
                                                     [R31, R32, R33]]))
    print("[INFO] :: Translation vector (scaled with d):", np.array([t1, t2, t3]))

    transformation_matrix_ir2w = np.array([
            [R11, R12, t1],
            [R21, R22, t2],
            [0, 0, 1]
        ])

    # load the images and apply the trasnformation to the infrared image
    ir_image_ir = cv2.imread(ir_image_file)
    height_ir, width_ir, _ = ir_image_ir.shape

    vi_image_wide = cv2.imread(wide_image_file)
    height_w, width_w, _ = vi_image_wide.shape


    transformed_ir_image = cv2.warpPerspective(ir_image_ir,
                                               transformation_matrix_ir2w,
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
    result_image = vi_image_wide.copy()
    result_image[top_left_y:top_left_y+height_cropped_ir, top_left_x:top_left_x+width_cropped_ir] = cropped_transformed_ir_image

    # Display the result
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Overlay infrared image on wide image')
    plt.show()

    ##########################
    # construct R' and t'
    R_z = []
    t_z = []

    # wide and infrared
    for (x1, y1), (x2, y2) in zip(ir_points_ir, vi_points_zoom):
        R_z.append([x1, y1, 1, 0, 0, 0, 0, 0, 0, -x2*x1, -x2*y1, -x2])
        R_z.append([0, 0, 0, x1, y1, 1, 0, 0, 0, -y2*x1, -y2*y1, -y2])
        R_z.append([0, 0, 0, 0, 0, 0, x1, y1, 1, -1*x1, -1*y1, -1])
        # Removing the row associated with z coordinate transformation
        # and setting R33 = 1, R31 = 0, R32 = 0, t3 = 0
        # This results in R31, R32, t3 not appearing in the linear system

        t_z.extend([x2-1/d, y2-1/d, 1-1/d])

    R_z = np.array(R_z)
    t_z = np.array(t_z)

    # solve for the parameters using least squares
    params, _, _, _ = np.linalg.lstsq(R_z, t_z, rcond=None)

    # extract the transformation matrix components
    # Manually set R31=R32=0, R33=1
    R11, R12, R13, R21, R22, R23, R31, R32, R33, t1, t2, t3 = params  

    print("[INFO] :: For infrared images to zoom images")
    print("[INFO] :: Transform matrix:\n", np.array([[R11, R12, R13], [R21, R22, R23],
                                                     [R31, R32, R33]]))
    print("[INFO] :: Translation vector (scaled with d):", np.array([t1, t2, t3]))

    transformation_matrix_ir2z = np.array([
            [R11, R12, t1],
            [R21, R22, t2],
            [0, 0, 1]
        ])

    ir_image_ir = cv2.imread(ir_image_file)
    height_ir, width_ir, _ = ir_image_ir.shape

    vi_image_zoom = cv2.imread(zoom_image_file)
    height_z, width_z, _ = vi_image_zoom.shape

    transformed_ir_image = cv2.warpPerspective(ir_image_ir,
                                               transformation_matrix_ir2z,
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
    top_left_x = (vi_image_zoom.shape[1] - width_cropped_ir) // 2
    top_left_y = (vi_image_zoom.shape[0] - height_cropped_ir) // 2

    # Overlay the cropped transformed infrared image onto the visible image
    result_image = vi_image_zoom.copy()
    result_image[top_left_y:top_left_y+height_cropped_ir, top_left_x:top_left_x+width_cropped_ir] = cropped_transformed_ir_image

    # Display the result
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('Overlay infrared image on zoom image')
    plt.show()


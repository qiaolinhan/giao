import cv2
import numpy as np

def register_images(visible_image_path, infrared_image_path):
    # Read images
    visible_img = cv2.imread(visible_image_path, cv2.IMREAD_GRAYSCALE)
    infrared_img = cv2.imread(infrared_image_path, cv2.IMREAD_GRAYSCALE)

    # Create SURF object
    surf = cv2.SURF_create()

    # Detect and compute keypoints and descriptors for visible image
    keypoints_visible, descriptors_visible = surf.detectAndCompute(visible_img, None)

    # Detect and compute keypoints and descriptors for infrared image
    keypoints_infrared, descriptors_infrared = surf.detectAndCompute(infrared_img, None)

    # Match keypoints
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_visible, descriptors_infrared, k=2)

    # Apply ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Draw matches
    img_matches = cv2.drawMatches(visible_img, keypoints_visible, infrared_img, keypoints_infrared, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show the matches
    cv2.imshow('Matches', img_matches)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Calculate the transformation matrix using the matched keypoints
    src_pts = np.float32([keypoints_visible[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_infrared[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Use RANSAC to estimate the transformation matrix
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Apply the transformation to align the images
    aligned_infrared_img = cv2.warpPerspective(infrared_img, M, (visible_img.shape[1], visible_img.shape[0]))

    # Display the aligned images
    cv2.imshow('Visible Image', visible_img)
    cv2.imshow('Aligned Infrared Image', aligned_infrared_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
Path = "/home/qiao/dev/datasets/original_datasets/FlightClub20220528"

vi_img_path = Path + "/m300_grabbed_data_1_17.1/rgb/0.png"
ir_img_path = Path + "/m300_grabbed_data_1_17.1/ir/0.png"

register_images(vi_img_path, ir_img_path)


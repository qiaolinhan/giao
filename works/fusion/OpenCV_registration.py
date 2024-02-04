
import cv2
import numpy as np

def image_registration(img_visible, img_infrared):
    # Convert images to grayscale
    gray_visible = cv2.cvtColor(img_visible, cv2.COLOR_BGR2GRAY)
    gray_infrared = cv2.cvtColor(img_infrared, cv2.COLOR_BGR2GRAY)

    # Detect ORB keypoints and descriptors
    orb = cv2.ORB_create(nfeatures=1000)
    keypoints_visible, descriptors_visible = orb.detectAndCompute(gray_visible, None)
    keypoints_infrared, descriptors_infrared = orb.detectAndCompute(gray_infrared, None)

    # Use BFMatcher to find the best matches between descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors_visible, descriptors_infrared, k=2)

    # Apply ratio test to get good matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) >= 4:
        H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # Rest of the code...
    else:
        print("[INFO]: Not enough matches to calculate homography.")


    # Draw matches
    img_matches = cv2.drawMatches(img_visible, keypoints_visible, img_infrared, keypoints_infrared, good_matches, None,
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Extract matched points
    src_pts = np.float32([keypoints_visible[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints_infrared[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Find homography matrix
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Apply homography to warp visible image onto the infrared image
    result = cv2.warpPerspective(img_visible, H, (img_infrared.shape[1], img_infrared.shape[0]))

    return result, H


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

registered_image, homography_matrix = image_registration(vi_original, ir_gray)

# Display the results
cv2.imshow('Visible Image', vi_original)
cv2.imshow('Infrared Image', ir_gray)
cv2.imshow('Registered Image', registered_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

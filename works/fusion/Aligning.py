
import cv2
import numpy as np

def align_visible_infrared_images(visible_image_path, infrared_image_path):
    # Read images
    visible_img = cv2.imread(visible_image_path, cv2.IMREAD_GRAYSCALE)
    infrared_img = cv2.imread(infrared_image_path, cv2.IMREAD_GRAYSCALE)

    # Perform histogram matching
    matched_infrared_img = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(infrared_img)

    # Display the original and matched infrared images for comparison
    cv2.imshow('Original Infrared Image', infrared_img)
    cv2.imshow('Matched Infrared Image', matched_infrared_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Find the transformation matrix using phase correlation
    shape = visible_img.shape
    half_height, half_width = shape[0] // 2, shape[1] // 2
    padded_visible_img = np.pad(visible_img, ((half_height, half_height), (half_width, half_width)), mode='constant')
    padded_infrared_img = np.pad(matched_infrared_img, ((half_height, half_height), (half_width, half_width)), mode='constant')

    fft_visible = np.fft.fft2(padded_visible_img)
    fft_infrared = np.fft.fft2(padded_infrared_img)

    cross_power_spectrum = fft_visible * np.conj(fft_infrared)
    cross_correlation = np.fft.ifft2(cross_power_spectrum / np.abs(cross_power_spectrum))

    # Find the peak (maximum correlation)
    y, x = np.unravel_index(np.argmax(cross_correlation), cross_correlation.shape)

    # Calculate the shift
    shift_x = x - half_width
    shift_y = y - half_height

    # Align the infrared image to the visible image using the shift
    aligned_infrared_img = np.roll(matched_infrared_img, (shift_y, shift_x), axis=(0, 1))

    # Display the aligned infrared image
    cv2.imshow('Aligned Infrared Image', aligned_infrared_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
align_visible_infrared_images('visible_image.jpg', 'infrared_image.jpg')

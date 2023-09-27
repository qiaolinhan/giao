import cv2
import numpy as np  

class OpenCV_fusion():
    def __init__(self, visible_img, infrared_img):
        self.visible_img = visible_img 
        self.infrared_img = infrared_img 

    def visible_transform(visible_img):
        # stretch the visible image so that the pixel level change:
        # 0-255 -> 0-65535
        visible_img = cv2.normalize(visible_img, None, 0, 65535, cv2.NORM_MINMAX)
        # transform the visible image pixel type into un-int 16 (np.uint16)
        visible_img = np.uint16(visible_img)

        # Gamma transform to enlarge the contrast of image 
        gamma = 2.2 
        visible_img = np.power((visible_img/float(np.max(visible_img))), gamma) 
        visible_t = np.uint16(visible_img * 65535)
        return visible_t

    def infrared_transform(infrared_img):
        # transform infrared image into 3 channel so that it could be conveniently
        # fused with visible image 
        infrared_t = cv2.cvtColor(infrared_img, cv2.COLOR_GRAY2BGR)
        return infrared_t

    def fusion_transform(visble_t, infrared_t):
        # weight visible image and infrared image and fuse them 
        alpha = 0.4
        beta = 1 - alpha 
        fusion_img = cv2.addWeighted(visible_t, alpha, infrared_t, beta, 0) 
        return fusion_img

if __name__ == "__main__":
    # loading the visible image and infrared image
    visible_img = cv2.imread("visible_img.png")
    infrared_img = cv2.imread("infrared_img.png")

    fused_img = OpenCV_fusion(visible_img, infrared_img)
    fused_img = np.transpose(fused_img.asnumpy(), (1, 2, 0))

    # print the info of fused_img
    # print(fused_img)
    # show result
    # cv2.imshow("fusion", fused_img)
    # cv2.waitkey(0)
    cv2.imwrite("fused_img.png", fused_img)

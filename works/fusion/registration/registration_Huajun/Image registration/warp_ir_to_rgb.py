import numpy as np
import cv2

# 读取红外和可见光图片
ir_image = cv2.imread('ir_test2.png')
rgb_image = cv2.imread('vs_test2.png')

# 手动输入的内参矩阵 Kir 和 Kvi
K_ir = np.array([
    [1044.03628677823, 0, 335.125645561794],
    [0, 1051.80215540345, 341.579677246452],
    [0, 0, 1]
])
K_vi = np.array([
    [2901.19910315714, 0, 940.239619965275],
    [0, 2893.75517626367, 618.475768281058],
    [0, 0, 1]
])

# 使用计算得出的旋转矩阵 R 和平移向量 t
# R = np.array([[1.99924680e+00, 1.32290932e-02, 1.76408589e+00],
#               [-5.04744627e-02, 2.02771208e+00, -1.96445012e+00],
#               [-6.80255440e-05, 3.12634272e-04, 7.37456230e-02]])
# t = np.array([-10.39379659, 11.13301228, 5.38164758])

# R_prime = np.array([[7.19448524e-01, 4.80912370e-03, -2.20637519e+02],
#                         [-1.83070465e-02, 7.37055695e-01, -2.33408904e+02],
#                         [1.06839427e-16, -2.35983091e-16, 1.03499298e+00]])
#
# t_prime = np.array([-8.69276816e+02, 4.97263404e+02, -2.03309209e-01])
# R = np.array([[ 0.99970638,  0.01842056,  0.01574256],
#        [-0.01805168,  0.99956643, -0.02326137],
#        [-0.01616422,  0.02297036,  0.99960546]])
# t = np.array([-0.16010031, -0.00981825, -2.89567168])

R = np.array([[1, 0, 0],
       [0, 1, 0],
       [0, 0, 1]])

# t = np.array([-1493,   -2891,  -58104])
t = np.array([-54.1240, -452.1266, -134.9375])




# 输入比例因子 d
d = float(input("Enter the scale factor d: "))

# # 计算逆矩阵
# R_inv = np.linalg.inv(R_prime)
# K_ir_inv = np.linalg.inv(K_p)

# 计算逆矩阵
R_inv = np.linalg.inv(R)
K_ir_inv = np.linalg.inv(K_ir)


# # 计算 R' 的逆矩阵
# R_prime_inv = np.linalg.inv(R_prime)


# # 计算映射矩阵
# M = R_prime_inv
#
# # 处理平移项，将其叠加到映射矩阵
# M[:, 2] -= (1 / d) * t_prime

# 计算映射矩阵
M = K_vi @ R_inv @ K_ir_inv

# 处理平移项，将其叠加到映射矩阵
M[:, 2] -= (1 / d) * K_vi @ R_inv @ t


# 使用 OpenCV 的 warpPerspective 进行透视变换
warped_ir_image = cv2.warpPerspective(ir_image, M, (rgb_image.shape[1], rgb_image.shape[0]))



# 调整透明度参数
alpha = 0.4  # 红外图像的透明度（0.0 - 1.0）
beta = 1 - alpha  # RGB 图像的透明度

# 叠加两个图像
blended_image = cv2.addWeighted(rgb_image, alpha, warped_ir_image, beta, 0)

# 显示结果
# cv2.imshow("Warped IR Image", warped_ir_image)
# cv2.imshow("RGB Image", rgb_image)
cv2.imshow("Blended Image", blended_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

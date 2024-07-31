import numpy as np
import cv2
import matplotlib.pyplot as plt

# 已拟合的 R 和 t
R = np.array([[ 0.99939278,  0.01788039, -0.02990576],
            [-0.02114781,  0.99339525, -0.11277701],
            [ 0.02769175,  0.11334097,  0.99317017]])
t = np.array([-8.28939947e-07, -2.66227653e-06, -9.72552714e-05])

# 手动输入内参矩阵
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

# 加载红外图像
ir_img_path = 'ir.png'
ir_img = cv2.imread(ir_img_path, cv2.IMREAD_GRAYSCALE)

# 创建 3x3 透视变换矩阵，将 (R | t) 填入
M = np.hstack((R, t.reshape(-1, 1)))

# 使用内参矩阵 K_ir 乘以组合矩阵 M
M_perspective = np.dot(K_ir, M).astype(np.float32)

# 只取前 3 列，作为 3x3 的矩阵传递给 warpPerspective
M_perspective = M_perspective[:, :3]

# 执行透视变换，将红外图像转换到 RGB 图像的坐标系中
h, w = ir_img.shape[:2]
warped_ir_img = cv2.warpPerspective(ir_img, M_perspective, (w, h))

# 可视化结果
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original IR Image')
plt.imshow(ir_img, cmap='gray')

plt.subplot(1, 2, 2)
plt.title('Warped IR Image')
plt.imshow(warped_ir_img, cmap='gray')

plt.show()

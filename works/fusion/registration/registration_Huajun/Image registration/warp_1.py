import cv2
import numpy as np
import matplotlib.pyplot as plt


def transform_ir_to_rgb(ir_image, Kir, Kvi, R, t):
    h, w = ir_image.shape[:2]

    # 构建齐次坐标的红外图像点矩阵
    y_indices, x_indices = np.indices((h, w))
    homogeneous_ir_coords = np.stack([x_indices.ravel(), y_indices.ravel(), np.ones_like(x_indices.ravel())])

    # 使用映射关系转换坐标
    Kvi_inv = np.linalg.inv(Kvi)
    transformed_coords = Kir @ (R @ (Kvi_inv @ homogeneous_ir_coords) + t[:, None])

    # 转换为非齐次坐标并四舍五入
    transformed_coords = transformed_coords[:2, :] / transformed_coords[2, :]
    transformed_coords = np.round(transformed_coords).astype(int)

    # 创建一个空白的目标图像
    warped_image = np.zeros((h, w), dtype=ir_image.dtype)

    # 遍历每个像素并映射到目标图像
    for i, (x, y) in enumerate(transformed_coords.T):
        if 0 <= x < w and 0 <= y < h:
            warped_image[y, x] = ir_image.ravel()[i]

    return warped_image


# 示例数据（需要替换为实际的 Kir、Kvi、R 和 t）
Kir = np.array([
    [1044.03628677823, 0, 335.125645561794],
    [0, 1051.80215540345, 341.579677246452],
    [0, 0, 1]
])
Kvi = np.array([
    [2901.19910315714, 0, 940.239619965275],
    [0, 2893.75517626367, 618.475768281058],
    [0, 0, 1]
])
R = np.array([
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1]
])

t = np.array([-0.0129, -0.0249, -0.4997])

# 读取红外图像
ir_image = cv2.imread('ir.png', cv2.IMREAD_GRAYSCALE)
rgb_image = cv2.imread('vs.png')

# 使用映射关系将红外图像映射到RGB图像坐标系
warped_ir_image = transform_ir_to_rgb(ir_image, Kir, Kvi, R, t)
print("RGB image shape:", rgb_image.shape)
print("Warped IR image shape:", warped_ir_image.shape)

# 显示两个图像并进行叠加
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# 显示 RGB 图像
axes[0].imshow(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
axes[0].set_title('RGB Image')
axes[0].axis('off')

# 显示红外图像
axes[1].imshow(warped_ir_image, cmap='gray')
axes[1].set_title('Warped IR Image')
axes[1].axis('off')

# 调整透明度参数
alpha = 0.3  # 红外图像的透明度（0.0 - 1.0）
beta = 1 - alpha  # RGB 图像的透明度

# 添加标题
plt.suptitle('Blended Image')

# 显示结果
plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()

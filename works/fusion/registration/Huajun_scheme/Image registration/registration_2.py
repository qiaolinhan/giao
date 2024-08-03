import numpy as np
import pandas as pd
from scipy.optimize import minimize
import cv2

# 从 Excel 文件读取数据
excel_data = pd.read_excel('manualPoints.xlsx')

# 获取可见光和热成像坐标
visible_points = excel_data[['Visible_X', 'Visible_Y']].head(6).values
thermal_points = excel_data[['Thermal_X', 'Thermal_Y']].head(6).values

# for point in visible_points:
#     q_vi = np.append(visible_points, 1)
# for point in thermal_points:
#     q_ir = np.append(thermal_points, 1)

# 假设你在表格中也有 `d` 的数据，并为它指定了列名
d = excel_data['d'].head(6).values

# 手动输入相机内参矩阵（举例）
K_ir = np.array([
    [1044.03628677823, 0, 335.125645561794],
    [0, 1051.80215540345, 341.579677246452],
    [0, 0, 1]])
K_vi = np.array([
    [2901.19910315714, 0, 940.239619965275],
    [0, 2893.75517626367, 618.475768281058],
    [0, 0, 1]])


def to_homogeneous(points):
    return np.hstack((points, np.ones((points.shape[0], 1))))

# 目标函数，计算残差
def residuals(params, visible_points, thermal_points, K_vi, K_ir):
    # 提取参数
    R_vec = params[:3]
    t = params[3:]

    # 构造旋转矩阵（使用 Rodrigues 公式）
    R, _ = cv2.Rodrigues(R_vec)

    # 将可见光坐标转换为红外坐标
    q_vi = to_homogeneous(visible_points)
    q_ir_pred = K_ir @ (R @ np.linalg.inv(K_vi) @ q_vi.T + np.reshape(t, (3, 1)) / d)
    q_ir_pred_normalized = q_ir_pred[:2] / q_ir_pred[2]
    q_ir_pred = q_ir_pred_normalized.T
    print(q_vi)

    print(q_ir_pred)
    print("\n")

    # 计算误差
    error = q_ir_pred - thermal_points
    return np.linalg.norm(error)


# 初始参数猜测（旋转矢量和平移向量）
initial_guess = np.zeros(6)

# 优化
result = minimize(
    residuals,
    initial_guess,
    args=(visible_points, thermal_points, K_vi, K_ir),
    method='L-BFGS-B'
)

# 提取结果
optimized_params = result.x
R_vec_optimized = optimized_params[:3]
t_optimized = optimized_params[3:]
R_optimized, _ = cv2.Rodrigues(R_vec_optimized)

print('优化后的旋转矩阵 R：')
print(R_optimized)
print('优化后的平移向量 t：')
print(t_optimized)




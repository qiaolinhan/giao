import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as R


# 将四元数转为旋转矩阵
def quaternion_to_rotation_matrix(quat):
    return R.from_quat(quat).as_matrix()

# 将旋转矩阵转为四元数
def rotation_matrix_to_quaternion(R_mat):
    return R.from_matrix(R_mat).as_quat()

# 目标函数
def residual(params, qir_data, qvi_data, Kir, Kvi, d_data):
    quat = params[:4]
    quat /= np.linalg.norm(quat)
    t = params[4:]
    R_matrix = quaternion_to_rotation_matrix(quat)

    residuals = []
    for qir, qvi, d in zip(qir_data, qvi_data, d_data):
        predicted_qir = Kir @ (R_matrix @ np.linalg.inv(Kvi) @ qvi) + (1 / d) * Kir @ t
        residuals.append(predicted_qir - qir)

    # 打印当前残差（调试用）
    print("Current residuals:", np.concatenate(residuals))
    return np.concatenate(residuals)


# 从 Excel 文件读取数据
excel_data = pd.read_excel('manualPoints.xlsx')

# 获取可见光和热成像坐标
visible_points = excel_data[['Visible_X', 'Visible_Y']].values
thermal_points = excel_data[['Thermal_X', 'Thermal_Y']].values
# 分离可见光和红外坐标的 X、Y 分量
vvi = visible_points[:, 0]
uvi = visible_points[:, 1]
vir = thermal_points[:, 0]
uir = thermal_points[:, 1]

# 假设你在表格中也有 `d` 的数据，并为它指定了列名
d = excel_data['d'].values

# 将两个一维数组拼接为一个二维数组
qvi_data = np.column_stack((uvi, vvi))

# 如果还需要一个额外的维度，比如 z 为 1（在 2D 平面上）
zvi = np.ones_like(uvi)  # 创建与 uvi 相同大小的全1数组作为 z 坐标

# 创建完整的3D二维数组
qvi_data = np.column_stack((uvi, vvi, zvi))

# 将两个一维数组拼接为一个二维数组
qir_data = np.column_stack((uir, vir))

# 如果还需要一个额外的维度，比如 z 为 1（在 2D 平面上）
zir = np.ones_like(uir)  # 创建与 uvi 相同大小的全1数组作为 z 坐标

# 创建完整的3D二维数组
qir_data = np.column_stack((uir, vir, zir))

# 假设你已知的 Kir 和 Kvi 矩阵

Kir = np.array([
    [1044.03628677823, 0, 335.125645561794],
    [0, 1051.80215540345, 341.579677246452],
    [0, 0, 1]
]) # 填入实际的 Kir 矩阵
Kvi = np.array([
    [2901.19910315714, 0, 940.239619965275],
    [0, 2893.75517626367, 618.475768281058],
    [0, 0, 1]
])  # 填入实际的 Kvi 矩阵

# 初始参数猜测
initial_quat = rotation_matrix_to_quaternion(np.eye(3))  # 初始为单位矩阵
initial_t = np.zeros(3)
initial_params = np.concatenate([initial_quat, initial_t])

# 使用最小二乘法进行优化
result = least_squares(residual, initial_params, args=(qir_data, qvi_data, Kir, Kvi, d))

# 获取最终的 R 和 t
final_quat = result.x[:4]
final_t = result.x[4:]
final_R = quaternion_to_rotation_matrix(final_quat)

print("Estimated R:", final_R)
print("Estimated t:", final_t)

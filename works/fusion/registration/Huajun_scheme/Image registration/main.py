import pandas as pd
import numpy as np
from scipy.optimize import least_squares
from scipy.optimize import minimize

# 从 Excel 文件读取数据
excel_data = pd.read_excel('./manualPoints.xlsx')

# 获取可见光和热成像坐标
visible_points = excel_data[['Visible_X', 'Visible_Y']].values
thermal_points = excel_data[['Thermal_X', 'Thermal_Y']].values
# 分离可见光和红外坐标的 X、Y 分量
uvi = visible_points[:, 0]
vvi = visible_points[:, 1]
uir = thermal_points[:, 0]
vir = thermal_points[:, 1]

# 假设你在表格中也有 `d` 的数据，并为它指定了列名
d_values = excel_data['d'].values

# 假设你已知的 Kir 和 Kvi 矩阵

K_ir = np.array([
    [1044.03628677823, 0, 335.125645561794],
    [0, 1051.80215540345, 341.579677246452],
    [0, 0, 1]
]) # 填入实际的 Kir 矩阵
K_vi = np.array([
    [2901.19910315714, 0, 940.239619965275],
    [0, 2893.75517626367, 618.475768281058],
    [0, 0, 1]
])  # 填入实际的 Kvi 矩阵

# 初始参数猜测 (随机或其他)
params_initial = np.random.randn(12)

# 残差函数
def residuals(params, visible_points, thermal_points, d_values):
    R = np.array(params[:9]).reshape(3, 3)
    t = np.array(params[9:])

    # R_prime = np.array(params[:9]).reshape(3, 3)
    # t_prime = np.array(params[9:])

    errors = []
    for (uvi, vvi), (uir, vir), d in zip(visible_points, thermal_points, d_values):
        q_vi = np.array([uvi, vvi, 1])
        q_ir_estimated = K_ir @ (R @ np.linalg.inv(K_vi) @ q_vi + (1 / d) * t)
        q_ir_actual = np.array([uir, vir, 1])
        # q_ir_estimated = R_prime @ q_vi + (1 / d) * t_prime
        # q_ir_actual = np.array([uir, vir, 1])

        errors.extend(q_ir_estimated - q_ir_actual)
    print("Current residuals:", errors)

    return np.array(errors)

# 使用最小二乘法进行优化
result = least_squares(residuals, params_initial, args=(visible_points, thermal_points, d_values))
# # 残差函数
# def residuals(params, uvi, vvi, uir, vir, d):
#     R = np.array(params[:9]).reshape(3, 3)
#     t = np.array(params[9:])
#
#     total_error = 0
#     # for u_vi, v_vi, u_ir, v_ir, d_i in zip(uvi, vvi, uir, vir, d):
#     #     source = np.array([u_vi, v_vi, 1])
#     #     target_estimated = R @ source + (1 / d_i) * t
#     #     target_actual = np.array([u_ir, v_ir, 1])
#
#     for uvi, vvi, uir, vir, d in zip(uvi, vvi, uir, vir, d):
#         source = np.array([uvi, vvi, 1])
#         qvi = np.array([uvi, vvi, 1])
#         target_estimated = K_ir @ (R @ np.linalg.inv(K_vi) @ qvi + (1 / d) * t)
#         target_actual = np.array([uir, vir, 1])
#     # q_vi = np.array([uvi, vvi, 1])
#     #         q_ir_estimated = K_ir @ (R @ np.linalg.inv(K_vi) @ q_vi + (1 / d) * t)
#     #         q_ir_actual = np.array([uir, vir, 1])
#     #         # q_ir_estimated = R_prime @ q_vi + (1 / d) * t_prime
#     #         # q_ir_actual = np.array([uir, vir, 1])
#
#         error = np.sum((target_estimated - target_actual) ** 2)
#         total_error += error
#
#     return total_error
#
# # 优化器函数
# def optimize_with_conjugate_gradient(params_initial, uvi, vvi, uir, vir, d):
#     result = minimize(residuals, params_initial, args=(uvi, vvi, uir, vir, d), method='CG')
#     return result.x
#
# # 使用共轭梯度法优化
# optimized_params = optimize_with_conjugate_gradient(params_initial, uvi, vvi, uir, vir, d)

# 获取优化后的 R 和 t 参数
optimized_params = result.x
R_optimized = optimized_params[:9].reshape(3, 3)
t_optimized = optimized_params[9:]

print("Optimized R:", R_optimized)
print("Optimized t:", t_optimized)

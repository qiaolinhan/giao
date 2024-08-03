% 从 Excel 文件中读取数据
filename = 'manualPoints.xlsx';
uvi = xlsread(filename, 'Visible_X'); % 替换为实际的列名称/范围
vvi = xlsread(filename, 'Visible_Y'); % 替换为实际的列名称/范围
uir = xlsread(filename, 'Thermal_X'); % 替换为实际的列名称/范围
vir = xlsread(filename, 'Thermal_Y'); % 替换为实际的列名称/范围

% 准备 q_vi 和 q_ir 的齐次坐标
num_points = length(uvi);
q_vi = [uvi, vvi, ones(num_points, 1)]';
q_ir = [uir, vir, ones(num_points, 1)]';

% 初始化优化参数
options = optimoptions('lsqnonlin', 'Display', 'iter');

% 初始值，假设 R 为单位矩阵，t 为零向量
initial_params = [reshape(eye(3), 9, 1); 0; 0; 0];

% 使用 lsqnonlin 进行非线性最小二乘优化
optimal_params = lsqnonlin(@reprojection_error, initial_params, [], [], options);

% 提取拟合的 R 和 t
R_opt = reshape(optimal_params(1:9), 3, 3);
t_opt = optimal_params(10:12);
disp('拟合的 R 矩阵：');
disp(R_opt);
disp('拟合的 t 向量：');
disp(t_opt);

% 在主代码块之后定义误差函数
function error = reprojection_error(params)
    % 提取 R 和 t
    R = reshape(params(1:9), 3, 3);
    t = params(10:12);
    
    % 投影变换
    global q_vi q_ir num_points
    q_vi_transformed = R * q_vi + repmat(t, 1, num_points);
    
    % 计算误差
    error = q_ir - q_vi_transformed;
    error = error(:); % 转换为列向量
end

% 主脚本内容
% 从 Excel 文件中导入数据
filename = 'manualPoints.xlsx';
visible_data = readtable(filename);
thermal_data = readtable(filename);
d_data = readtable(filename);

% 获取坐标并转化为齐次坐标
visible_points = [visible_data.Visible_X, visible_data.Visible_Y, ones(height(visible_data), 1)];
thermal_points = [thermal_data.Thermal_X, thermal_data.Thermal_Y, ones(height(thermal_data), 1)];
d = [d_data.d];

% 初始旋转矢量和平移向量猜测（零值或小随机数）
initial_guess = 0.01 * randn(1, 12);

% 优化设置
options = optimoptions('lsqnonlin', 'Display', 'iter', 'Algorithm', 'trust-region-reflective');
[result, ~] = lsqnonlin(@(params) compute_residuals(params, visible_points, thermal_points, d), initial_guess, [], [], options);


% 提取最终优化结果
R_vec_optimized = result(1:9);
R_optimized = reshape(R_vec_optimized, [3, 3]);
t_optimized = result(10:12).';

% 显示优化后的旋转和平移矩阵
disp('优化后的旋转矩阵 R：');
disp(R_optimized);
disp('优化后的平移向量 t：');
disp(t_optimized);



function error = compute_residuals(params, visible_points, thermal_points, d)
    % 提取参数
    R_vec = params(1:9);  % 9 个元素，3x3 矩阵
    R = reshape(R_vec, [3, 3]);
    t = params(10:12);

    % 将可见光坐标转换为红外坐标
    q_ir_pred = R * visible_points.' + (t(:) ./ d.');

    % 归一化齐次坐标
    q_ir_pred = q_ir_pred(1:2, :) ./ q_ir_pred(3, :);

    % 计算误差
    error = sqrt(sum((q_ir_pred.' - thermal_points(:, 1:2)).^2, 2));
    q_ir_pred = q_ir_pred(1:2, :)';
    disp(q_ir_pred);
end
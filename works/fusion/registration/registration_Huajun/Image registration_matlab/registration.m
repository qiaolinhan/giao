% Main script content
% Import data from an Excel file
filename = 'manualPoints.xlsx';
visible_data = readtable(filename);
thermal_data = readtable(filename);
d_data = readtable(filename);

% Get coordinates and convert them to homogeneous coordinates
visible_points = [visible_data.Visible_X, visible_data.Visible_Y, ones(height(visible_data), 1)];
thermal_points = [thermal_data.Thermal_X, thermal_data.Thermal_Y, ones(height(thermal_data), 1)];
d = [d_data.d];

% Manually input the camera intrinsic matrices
% K_vi = [2901.19910315714, 0, 940.239619965275;
%         0, 2893.75517626367, 618.475768281058;
%         0, 0, 1];

K_vi = [1484.39712549035,	0,	964.013870831680;
        0,	1478.75546438438,	577.707036666613;
        0,	0,	1];
K_ir = [1044.03628677823, 0, 335.125645561794;
        0, 1051.80215540345, 341.579677246452;
        0, 0, 1];

% Initial guesses for rotation vector and translation vector (zeros or small random numbers)
initial_guess = zeros(1, 6);

% Optimization settings
options = optimoptions('lsqnonlin', 'Display', 'iter', 'Algorithm', 'trust-region-reflective');
[result, ~] = lsqnonlin(@(params) compute_residuals(params, visible_points, thermal_points, K_vi, K_ir, d), initial_guess, [], [], options);

% Extract the final optimized result
R_vec_optimized = result(1:3);
t_optimized = result(4:6);
R_optimized = rotationVectorToMatrix(R_vec_optimized);

% Display the optimized rotation and translation matrices
disp('Optimized rotation matrix R:');
disp(R_optimized);
disp('Optimized translation vector t:');
disp(t_optimized);

% Helper function: Compute residuals
function error = compute_residuals(params, visible_points, thermal_points, K_vi, K_ir, d)
    % Extract parameters
    R_vec = params(1:3);
    t = params(4:6);

    % Convert the rotation vector into a rotation matrix
    R = rotationVectorToMatrix(R_vec);
    R_inv = inv(R);

    % Calculate the mapping matrix M
    M = K_vi * R_inv * inv(K_ir);
    
    % Convert `t` to a column vector
    t_column = reshape(t, 3, 1);
    
    % Ensure `d` is a scalar value
    scale_factor = 1 / d(1);

    % Compute the translation component
    M(:, 3) = M(:, 3) - scale_factor * K_vi * R_inv * t_column;

    % Transform infrared coordinates to visible light coordinates
    q_ir = [thermal_points(:, 1:2), ones(size(thermal_points, 1), 1)]';
    q_vi_pred = M * q_ir;

    % Normalize homogeneous coordinates
    q_vi_pred(1, :) = q_vi_pred(1, :) ./ q_vi_pred(3, :);
    q_vi_pred(2, :) = q_vi_pred(2, :) ./ q_vi_pred(3, :);

    % Transpose to maintain original matrix structure
    q_vi_pred = q_vi_pred(1:2, :)';
    disp(q_vi_pred);
    disp(d)
    
    % Compute error
    error = sqrt(sum((q_vi_pred - visible_points(:, 1:2)).^2, 2));
end

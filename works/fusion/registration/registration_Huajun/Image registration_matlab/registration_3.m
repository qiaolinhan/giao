% Step 1: Load data
filename = 'manualPoints.xlsx';
visible_data = readtable(filename);
thermal_data = readtable(filename);
d_data = readtable(filename);

% Get coordinates and convert to homogeneous coordinates
qvi = [visible_data.Visible_X, visible_data.Visible_Y, ones(height(visible_data), 1)];
qir = [thermal_data.Thermal_X, thermal_data.Thermal_Y, ones(height(thermal_data), 1)];
d = [d_data.d];

% Step 2: Define the initial guesses for R' and t'
R_guess = eye(3);
t_guess = zeros(3, 1);
params_guess = [R_guess(:); t_guess];

% Step 4: Use lsqnonlin for optimization
options = optimoptions('lsqnonlin', 'Display', 'iter', 'Algorithm', 'trust-region-reflective');
params_optimized = lsqnonlin(@(params) transformation_error(params, qvi, qir, d), params_guess, [], [], options);

% Step 5: Extract the optimized R' and t'
R_optimized = reshape(params_optimized(1:9), 3, 3);
t_optimized = params_optimized(10:12);

% Display results
disp('Optimized R matrix:');
disp(R_optimized);
disp('Optimized t vector:');
disp(t_optimized);

function residuals = transformation_error(params, qvi, qir, d_data)
    % Extract R and t
    R_vec = params(1:9);
    t_vec = params(10:12);

    R_matrix = reshape(R_vec, 3, 3);
    t_vector = t_vec(:); % Ensure it's a column vector

    % Initialize residuals
    num_points = size(qvi, 1); % Assuming qvi is N x 3
    residuals = zeros(3, num_points);

    % Compute residuals for each point
    for i = 1:num_points
        % Calculate estimated q_ir as a column vector
        qir_estimated = R_matrix * qvi(i, :)' + (1/d_data(i)) * t_vector;
        % Compute residuals as column vectors
        residuals(:, i) = qir(i, :)' - qir_estimated;
        disp(qir_estimated)
    end

    % Vectorize residuals for lsqnonlin
    residuals = residuals(:);
end

% MATLAB Script to Compare Occupation Kernel (OK) vs. Regularized Motion
% Tomography (RMT) for a Sparse Sensing Scenario.
%
% OBJECTIVE:
% To demonstrate that the Occupation Kernel method is more accurate and 
% computationally efficient than a grid-based RMT method when a small 
% number of trajectories sample a large domain with a localized flow field.

clear; clc; close all;
rng(42); % For reproducibility

%% 1. EXPERIMENT PARAMETERS
% --- Domain and Grid ---
domain_size = 10;
grid_res = 100; % Grid resolution (P = grid_res^2)
x_grid = linspace(0, domain_size, grid_res);
y_grid = linspace(0, domain_size, grid_res);
[X, Y] = meshgrid(x_grid, y_grid);
P = grid_res^2; % Total number of grid points

% --- Trajectory and Simulation ---
M = 15; % Number of trajectories (M << P)
T_final = 10; % Final simulation time
sensor_speed = 1.0;

% --- Algorithm Parameters ---
lambda = 0.1;  % Regularization parameter for RMT
sigma = 0.5;   % Width of the Gaussian kernel for OK

%% 2. GROUND TRUTH FLOW FIELD (Localized Vortex Pair)
fprintf('Step 1: Defining ground truth flow field...\n');
true_flow_func = @(x,y) vortex_pair_flow(x, y);
[U_true, V_true] = true_flow_func(X, Y);

%% 3. DATA GENERATION (Simulate Trajectories)
fprintf('Step 2: Generating sparse trajectory data (M=%d)...\n', M);
start_points = [zeros(M, 1), domain_size * rand(M, 1)];
start_angles = (rand(M, 1) - 0.5) * (pi / 3); % Angles between -30 and +30 deg

true_trajectories = cell(M, 1);
final_points = zeros(M, 2);
ode_options = odeset('RelTol', 1e-6, 'AbsTol', 1e-6);

for i = 1:M
    v_s = sensor_speed * [cos(start_angles(i)), sin(start_angles(i))];
    ode_func = @(t, pos) v_s' + true_flow_func(pos(1), pos(2));
    [~, traj] = ode45(ode_func, [0, T_final], start_points(i,:), ode_options);
    true_trajectories{i} = traj;
    final_points(i,:) = traj(end,:);
end

% Data available to the algorithms: start_points, start_angles, final_points
anticipated_final_points = start_points + sensor_speed * [cos(start_angles), sin(start_angles)] * T_final;
displacement_error = final_points - anticipated_final_points;

%% 4. ALGORITHM 1: OCCUPATION KERNEL (OK) METHOD
fprintf('Step 3: Running Occupation Kernel (OK) Method...\n');
tic;

% Define basis functions (Gaussian tubes around straight paths)
Gamma_flat = zeros(P, M);
for i = 1:M
    p1 = start_points(i,:);
    p2 = anticipated_final_points(i,:);
    dist_sq = distance_to_segment_sq(X, Y, p1, p2);
    gamma_i = exp(-dist_sq / (2 * sigma^2));
    Gamma_flat(:,i) = gamma_i(:);
end

% Build the M-by-M Gram matrix 'A'
A = zeros(M, M);
num_path_points = 100;
for i = 1:M
    path_x = linspace(start_points(i,1), anticipated_final_points(i,1), num_path_points);
    path_y = linspace(start_points(i,2), anticipated_final_points(i,2), num_path_points);
    path_length = norm(anticipated_final_points(i,:) - start_points(i,:));
    
    for j = 1:M
        gamma_j = reshape(Gamma_flat(:,j), grid_res, grid_res);
        gamma_j_on_path = interp2(X, Y, gamma_j, path_x, path_y, 'linear', 0);
        A(i,j) = sum(gamma_j_on_path) * (path_length / num_path_points);
    end
end

% Solve for weights and reconstruct the flow field
if rank(A) < M
    warning('OK system matrix is singular. Using pseudoinverse.');
    weights = pinv(A) * displacement_error;
else
    weights = A \ displacement_error;
end
F_OK_flat = Gamma_flat * weights;
U_OK = reshape(F_OK_flat(:,1), grid_res, grid_res);
V_OK = reshape(F_OK_flat(:,2), grid_res, grid_res);
time_OK = toc;
fprintf('...OK Method finished in %.4f seconds.\n', time_OK);

%% 5. ALGORITHM 2: REGULARIZED MOTION TOMOGRAPHY (RMT) METHOD
fprintf('Step 4: Running RMT Method...\n');
tic;

% Create the discrete Laplacian operator R (P-by-P sparse matrix)
R = create_laplacian(grid_res, grid_res);

% Create the sampling/integration matrix T (M-by-P sparse matrix)
T_matrix = sparse(M, P);
for i = 1:M
    p1 = start_points(i,:);
    p2 = anticipated_final_points(i,:);
    path_len = norm(p2 - p1);
    [line_indices, ~] = bresenham(p1, p2, x_grid, y_grid);
    if ~isempty(line_indices)
        T_matrix(i, line_indices) = path_len / length(line_indices);
    end
end

% Construct and solve the large P-by-P RMT system
A_rmt = T_matrix' * T_matrix + lambda * (R' * R);
F_RMT_u_flat = A_rmt \ (T_matrix' * displacement_error(:,1));
F_RMT_v_flat = A_rmt \ (T_matrix' * displacement_error(:,2));

% Reshape flow field back to 2D grid
U_RMT = reshape(F_RMT_u_flat, grid_res, grid_res);
V_RMT = reshape(F_RMT_v_flat, grid_res, grid_res);
time_RMT = toc;
fprintf('...RMT Method finished in %.4f seconds.\n', time_RMT);

%% 6. EVALUATION
fprintf('Step 5: Evaluating results...\n');

% Global Mean Squared Error (MSE)
mse_global_OK = mean((U_OK(:) - U_true(:)).^2 + (V_OK(:) - V_true(:)).^2);
mse_global_RMT = mean((U_RMT(:) - U_true(:)).^2 + (V_RMT(:) - V_true(:)).^2);

% On-Trajectory MSE
error_sum_OK = 0; error_sum_RMT = 0; total_path_points = 0;
for i = 1:M
    traj = true_trajectories{i};
    num_pts = size(traj, 1);
    
    U_true_path = interp2(X, Y, U_true, traj(:,1), traj(:,2), 'linear', 0);
    V_true_path = interp2(X, Y, V_true, traj(:,1), traj(:,2), 'linear', 0);
    
    U_est_OK_path = interp2(X, Y, U_OK, traj(:,1), traj(:,2), 'linear', 0);
    V_est_OK_path = interp2(X, Y, V_OK, traj(:,1), traj(:,2), 'linear', 0);
    
    U_est_RMT_path = interp2(X, Y, U_RMT, traj(:,1), traj(:,2), 'linear', 0);
    V_est_RMT_path = interp2(X, Y, V_RMT, traj(:,1), traj(:,2), 'linear', 0);
    
    error_sum_OK = error_sum_OK + sum((U_est_OK_path - U_true_path).^2 + (V_est_OK_path - V_true_path).^2);
    error_sum_RMT = error_sum_RMT + sum((U_est_RMT_path - U_true_path).^2 + (V_est_RMT_path - V_true_path).^2);
    total_path_points = total_path_points + num_pts;
end
mse_ontraj_OK = error_sum_OK / total_path_points;
mse_ontraj_RMT = error_sum_RMT / total_path_points;

%% 7. VISUALIZATION
fprintf('Step 6: Generating plots...\n');

quiver_skip = 5; % Downsample for cleaner quiver plots

% --- Figure 1: Visual Comparison of Flow Fields ---
figure('Position', [100, 100, 1600, 400], 'Name', 'Visual Comparison');

% Subplot 1: Ground Truth
subplot(1, 4, 1);
quiver(X(1:quiver_skip:end, 1:quiver_skip:end), Y(1:quiver_skip:end, 1:quiver_skip:end), ...
       U_true(1:quiver_skip:end, 1:quiver_skip:end), V_true(1:quiver_skip:end, 1:quiver_skip:end), 1.5);
hold on;
for i = 1:M
    plot(true_trajectories{i}(:,1), true_trajectories{i}(:,2), 'r-', 'LineWidth', 1.5);
end
hold off;
axis equal; axis([0 domain_size 0 domain_size]);
title({'1. Ground Truth Flow Field', 'with True Trajectories'});
xlabel('x'); ylabel('y');

% Subplot 2: OK Reconstruction
subplot(1, 4, 2);
quiver(X(1:quiver_skip:end, 1:quiver_skip:end), Y(1:quiver_skip:end, 1:quiver_skip:end), ...
       U_OK(1:quiver_skip:end, 1:quiver_skip:end), V_OK(1:quiver_skip:end, 1:quiver_skip:end), 1.5);
axis equal; axis([0 domain_size 0 domain_size]);
title({'2. Occupation Kernel (OK) Reconstruction', sprintf('Time: %.4f s', time_OK)});
xlabel('x');

% Subplot 3: RMT Reconstruction
subplot(1, 4, 3);
quiver(X(1:quiver_skip:end, 1:quiver_skip:end), Y(1:quiver_skip:end, 1:quiver_skip:end), ...
       U_RMT(1:quiver_skip:end, 1:quiver_skip:end), V_RMT(1:quiver_skip:end, 1:quiver_skip:end), 1.5);
axis equal; axis([0 domain_size 0 domain_size]);
title({'3. RMT Reconstruction', sprintf('Time: %.4f s', time_RMT)});
xlabel('x');

% Subplot 4: Error Comparison
subplot(1, 4, 4);
err_OK_mag = sqrt((U_OK - U_true).^2 + (V_OK - V_true).^2);
err_RMT_mag = sqrt((U_RMT - U_true).^2 + (V_RMT - V_true).^2);
pcolor(X, Y, err_RMT_mag - err_OK_mag); % Positive (blue) means RMT error is higher
shading interp;
colorbar;
axis equal; axis([0 domain_size 0 domain_size]);
title({'4. Error Difference (RMT - OK)', 'Blue = RMT has higher error'});
xlabel('x');

% --- Figure 2: Quantitative Comparison ---
figure('Position', [200, 200, 800, 400], 'Name', 'Quantitative Comparison');
categories = {'OK', 'RMT'};

% Subplot 1: MSE Comparison
subplot(1, 2, 1);
bar_data = [mse_global_OK, mse_global_RMT; mse_ontraj_OK, mse_ontraj_RMT];
b = bar(bar_data);
set(gca, 'XTickLabel', {'Global MSE', 'On-Trajectory MSE'});
ylabel('Mean Squared Error');
legend(categories);
title('Error Metrics');
grid on;

% Subplot 2: Timing Comparison
subplot(1, 2, 2);
bar([time_OK, time_RMT]);
set(gca, 'XTickLabel', categories);
ylabel('Execution Time (seconds)');
title('Computational Cost');
grid on;
fprintf('...Done.\n');




% --- Figure 3: Absolute Error Heatmaps ---
figure('Position', [300, 100, 1000, 400], 'Name', 'Absolute Error Comparison');

% Calculate error magnitudes
err_OK_mag = sqrt((U_OK - U_true).^2 + (V_OK - V_true).^2);
err_RMT_mag = sqrt((U_RMT - U_true).^2 + (V_RMT - V_true).^2);

% Determine a common color scale for fair comparison

max_err = max([err_OK_mag(:); err_RMT_mag(:)]); % Find the single overall maximum

% Subplot 1: Occupation Kernel Absolute Error
subplot(1, 2, 1);
pcolor(X, Y, err_OK_mag);
shading interp;
axis equal; axis([0 domain_size 0 domain_size]);
colorbar;
caxis([0 max_err]); % Use common color axis
hold on;
for i = 1:M % Overlay trajectories to see context
    plot(true_trajectories{i}(:,1), true_trajectories{i}(:,2), 'k-', 'LineWidth', 0.5);
end
hold off;
title({'Absolute Error Magnitude', 'Occupation Kernel (OK)'});
xlabel('x'); ylabel('y');

% Subplot 2: RMT Absolute Error
subplot(1, 2, 2);
pcolor(X, Y, err_RMT_mag);
shading interp;
axis equal; axis([0 domain_size 0 domain_size]);
colorbar;
caxis([0 max_err]); % Use common color axis
hold on;
for i = 1:M
    plot(true_trajectories{i}(:,1), true_trajectories{i}(:,2), 'k-', 'LineWidth', 0.5);
end
hold off;
title({'Absolute Error Magnitude', 'RMT'});
xlabel('x'); ylabel('y');


% --- Figure 4: Error Vector Fields ---
figure('Position', [400, 100, 1000, 400], 'Name', 'Error Vector Comparison');

% Calculate error vectors
err_U_OK = U_OK - U_true;
err_V_OK = V_OK - V_true;
err_U_RMT = U_RMT - U_true;
err_V_RMT = V_RMT - V_true;

% Subplot 1: OK Error Field
subplot(1, 2, 1);
quiver(X(1:quiver_skip:end, 1:quiver_skip:end), Y(1:quiver_skip:end, 1:quiver_skip:end), ...
       err_U_OK(1:quiver_skip:end, 1:quiver_skip:end), err_V_OK(1:quiver_skip:end, 1:quiver_skip:end), 1.5);
axis equal; axis([0 domain_size 0 domain_size]);
title({'Error Vector Field (F_{est} - F_{true})', 'Occupation Kernel (OK)'});
xlabel('x'); ylabel('y');

% Subplot 2: RMT Error Field
subplot(1, 2, 2);
quiver(X(1:quiver_skip:end, 1:quiver_skip:end), Y(1:quiver_skip:end, 1:quiver_skip:end), ...
       err_U_RMT(1:quiver_skip:end, 1:quiver_skip:end), err_V_RMT(1:quiver_skip:end, 1:quiver_skip:end), 1.5);
axis equal; axis([0 domain_size 0 domain_size]);
title({'Error Vector Field (F_{est} - F_{true})', 'RMT'});
xlabel('x'); ylabel('y');


% --- Figure 5: Statistical Error Distribution ---
figure('Position', [500, 200, 600, 400], 'Name', 'Error Distribution');

histogram(err_OK_mag(:), 50, 'Normalization', 'probability', 'DisplayName', 'OK');
hold on;
histogram(err_RMT_mag(:), 50, 'Normalization', 'probability', 'DisplayName', 'RMT');
hold off;
xlabel('Error Magnitude');
ylabel('Probability Density');
title('Statistical Distribution of Pointwise Errors');
legend;
grid on;

%% HELPER FUNCTIONS

function [u, v] = vortex_pair_flow(x, y)
    % Defines a localized flow field with two counter-rotating vortices.
    strength = 1.0;
    radius = 1.5;
    c1_x = 5; c1_y = 7.5; % Center of vortex 1
    c2_x = 5; c2_y = 2.5; % Center of vortex 2

    % Vortex 1 (clockwise)
    r1_sq = (x - c1_x).^2 + (y - c1_y).^2;
    exp_term1 = exp(-r1_sq / (2 * radius^2));
    u1 = strength * (y - c1_y) .* exp_term1;
    v1 = -strength * (x - c1_x) .* exp_term1;

    % Vortex 2 (counter-clockwise)
    r2_sq = (x - c2_x).^2 + (y - c2_y).^2;
    exp_term2 = exp(-r2_sq / (2 * radius^2));
    u2 = -strength * (y - c2_y) .* exp_term2;
    v2 = strength * (x - c2_x) .* exp_term2;

    u = u1 + u2;
    v = v1 + v2;
end

function dist_sq = distance_to_segment_sq(px, py, p1, p2)
    % Calculates the squared distance from grid points (px, py) to a line segment p1-p2.
    L_sq = (p2(1)-p1(1))^2 + (p2(2)-p1(2))^2;
    if L_sq == 0 % If p1 and p2 are the same point
        dist_sq = (px-p1(1)).^2 + (py-p1(2)).^2;
        return;
    end
    % Project points onto the line defined by the segment
    t = ((px-p1(1))*(p2(1)-p1(1)) + (py-p1(2))*(p2(2)-p1(2))) / L_sq;
    t = max(0, min(1, t)); % Clamp projection to be within the segment [0, 1]
    
    proj_x = p1(1) + t.*(p2(1)-p1(1));
    proj_y = p1(2) + t.*(p2(2)-p1(2));
    
    dist_sq = (px-proj_x).^2 + (py-proj_y).^2;
end

function R = create_laplacian(nx, ny)
    % Creates a sparse PxP matrix for the 2D Laplacian operator using a 
    % 5-point stencil with Neumann boundary conditions (zero flux).
    N = nx * ny;
    e = ones(N, 1);
    % Standard 5-point stencil for the interior
    R = spdiags([e e -4*e e e], [-nx -1 0 1 nx], N, N);
    
    % Adjust boundaries to enforce Neumann conditions
    % This is done by modifying the stencil for boundary nodes.
    for i = 1:nx % Iterate over columns
        % Bottom boundary points in this column
        k = i;
        if k+ny <= N, R(k, k+ny) = 2; end
        
        % Top boundary points in this column
        k = N - nx + i;
        if k-ny >= 1, R(k, k-ny) = 2; end
    end
    for i = 1:ny % Iterate over rows
        % Left boundary points in this row
        k = (i-1)*nx + 1;
        if k+1 <= N, R(k, k+1) = 2; end

        % Right boundary points in this row
        k = i*nx;
        if k-1 >= 1, R(k, k-1) = 2; end
    end
end

function [indices, coords] = bresenham(p1, p2, x_grid, y_grid)
    % Finds grid cell indices intersected by a line segment p1-p2 using
    % Bresenham's line algorithm.
    nx = length(x_grid); ny = length(y_grid);
    dx = x_grid(2)-x_grid(1); dy = y_grid(2)-y_grid(1);

    % Convert world coordinates to grid coordinates
    x1 = round((p1(1) - x_grid(1))/dx) + 1;
    y1 = round((p1(2) - y_grid(1))/dy) + 1;
    x2 = round((p2(1) - x_grid(1))/dx) + 1;
    y2 = round((p2(2) - y_grid(1))/dy) + 1;
    
    % Clamp to grid boundaries
    x1 = max(1, min(nx, x1)); y1 = max(1, min(ny, y1));
    x2 = max(1, min(nx, x2)); y2 = max(1, min(ny, y2));

    % Standard Bresenham's algorithm implementation
    steep = abs(y2 - y1) > abs(x2 - x1);
    if steep, [x1, y1] = swap(x1, y1); [x2, y2] = swap(x2, y2); end
    if x1 > x2, [x1, x2] = swap(x1, x2); [y1, y2] = swap(y1, y2); end
    
    delx = x2 - x1;
    dely = abs(y2 - y1);
    error = 0;
    y = y1;
    if y1 < y2, ystep = 1; else, ystep = -1; end
    
    coords = [];
    for x = x1:x2
        if steep
            coords = [coords; y, x]; %#ok<AGROW>
        else
            coords = [coords; x, y]; %#ok<AGROW>
        end
        error = error + dely;
        if 2*error >= delx
            y = y + ystep;
            error = error - delx;
        end
    end
    % Convert 2D grid coordinates to 1D linear indices
    indices = sub2ind([ny, nx], coords(:,2), coords(:,1));
end

function [b, a] = swap(a, b)
    % Simple utility to swap two variables.
end

%% Problem 5
A = [1 2 -4 5; 3 6 -2 1; 2 4 -3 3];
[M, N] = size(A);

%% Part a, SVD applied

[U, sigma, V_t] = svd(A); % U 3x3, sigma 3x4, V_t 4x4
V = V_t'; % V 4x4
p = rank(A); % 2

[Up, U0] = deal(U(:, 1:p), U(:, p+1:end));
[Vp, V0] = deal(V(:, 1:p), V(:, p+1:end));
sigma_p = sigma(1:p, 1:p); % s 2x2
A_reduced = Up * sigma_p * Vp';

Up_check = Up' * Up;
Vp_check = Vp' * Vp;

U0_check = U0' * Up;
V0_check = V0' * Vp;

%% Part b, least squares
y = [2 1 4]';

A_pseudo = V_t * pinv(sigma) * U';

x_ls1 = pinv(A) * y;
x_ls2 = A_pseudo * y;

residual_n1 = norm(y - A*x_ls1, 2)^2;
residual_n2 = norm(y - A*x_ls2, 2)^2;

%% 






















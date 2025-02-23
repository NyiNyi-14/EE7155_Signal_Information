%% a, A creation
load("blocksdeconv.mat");
A = convolution_matrix(h, x);

%% b, SVD of A
[U, sigma, V_t] = svd(A);

[largest_sig, pos_1] = max(sigma(:));
[smallest_sig, pos_2] = min(sigma(:));
% [row_1, column_1] = ind2sub(size(sigma), pos_1);
% [row_2, column_2] = ind2sub(size(sigma), pos_2);

A_pseudo = V_t * pinv(sigma) * U';

x_pinv = A_pseudo * y;

% plotted in the last cell, compare with the rest of the problem

%% c, pinv(A) to noise
x_ls = A_pseudo * yn;
mean_sq_E = norm(x - x_ls, 2)^2;

% plotted in the last cell, compare with the rest of the problem

%% d, truncated SVD
q = 10:1:300;
p = rank(A);
sigma_trun = sigma;
recon_error = zeros(1, length(q));

for i = 1:length(q)
    sigma_trun(:, p-q(i):end) = 0;
    A_trun = U * sigma_trun * V_t;
    A_trun_pseu = V_t * pinv(sigma_trun) * U';
    x_SCD_trun = A_trun_pseu * yn;
    recon_error(i) = norm(x_SCD_trun - x, 2)^2;

end

figure (1);
clf;
plot(q, recon_error);
xlabel("q, truncated values");
ylabel("Reconstruction error");
title("Reconstruction error with respect to q");
grid on;

[SVD_trun_E, idx_SVD] = min(recon_error);
best_q = q(idx_SVD);
fprintf("Lowest recon error is %.4f with the q value of %.1f \n", SVD_trun_E, best_q);

% plotted in the last cell, compare with the rest of the problem

%% e, Tikhonov regularization
delta = 1e-4:0.0001:0.01;
recon_E_tik = zeros(1, length(delta));

for j = 1:length(delta)
    x_tik = (A'*A + delta(j)*eye(size(A,2))) \ (A' * yn);
    recon_E_tik(j) = norm(x_tik - x, 2)^2;
end

figure (2);
clf;
plot(delta, recon_E_tik);
xlabel("delta values");
ylabel("Reconstruction error");
title("Reconstruction error with respect to delta");
grid on;

[Tik_E, idx_tik] = min(recon_E_tik);
best_delta = delta(idx_tik);

fprintf("Lowest recon error with Tik is %.4f with delta value of %.4f \n", Tik_E, best_delta);

x_tik_best = (A'*A + best_delta*eye(size(A,2))) \ (A' * yn);

% plotted in the last cell, compare with the rest of the problem

%% f, comparison of the above
base = norm(x - yn(1: length(x), :), 2)^2;

figure (3);
clf;
bar([mean_sq_E, SVD_trun_E, Tik_E, base]);
set(gca, 'XTickLabel', {'Mean Square', 'SVD Trun', 'Tik', 'Base'});
ylabel('Error Magnitude');
title('Comparison of Errors');
grid on;

figure (4);
clf;

subplot(2,2,1);
plot(x_pinv);
xlabel("index");
ylabel("Values");
title('Projection to noise free');
grid on;

subplot(2,2,2);
plot(x_ls);
xlabel("index");
ylabel("Noisy Values");
title('Projection to noisy');
grid on;

subplot(2,2,3);
plot(x_SCD_trun);
xlabel("SVD x index");
ylabel("Noisy Values");
title('SVD truncated to noisy');
grid on;

subplot(2,2,4);
plot(x_tik_best);
xlabel("Tik x index");
ylabel("Noisy Values");
title('Tik to noisy');
grid on;

%% 






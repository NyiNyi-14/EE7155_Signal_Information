%% Nyi Nyi Aung_Coursework 3
%
clear; clc; close all;

%% (A) Define the true system H(z) and generate impulse response h(n)

b_true = [1, -0.92, 0.81];                
a_true = [1, -1.978, 2.853, -1.877, 0.9036]; 
N = 100;   
h = impz(b_true, a_true, N);  % h(n)

figure; stem(h, 'filled');
title('True Impulse Response h(n)');
xlabel('n'); ylabel('h(n)');
grid on;

%% (B) Iterative Prefiltering

p = 4;  % 4-pole
q = 2;  % 2-zero

[b_est, a_est] = stmcb(h, q, p);

disp('--- (B) Iterative Prefiltering, ARMA(2,4) results ---');
disp('Estimated numerator b_est ='); disp(b_est);
disp('Estimated denominator a_est ='); disp(a_est);

% the impulse response of the fitted model:
h_est = impz(b_est, a_est, N);

mse_smc = mean((h - h_est).^2); % MSE
disp(['MSE of ARMA(2,4) fit via stmcb: ', num2str(mse_smc)]);

%% Over-estimating with 4-zero, 4-pole

p_ov4 = 4; 
q_ov4 = 4;
[b_est_ov4, a_est_ov4] = stmcb(h, q_ov4, p_ov4);
h_est_ov4 = impz(b_est_ov4, a_est_ov4, N);
mse_ov4 = mean((h - h_est_ov4).^2);

disp('--- Over-specified ARMA(4,4) via stmcb ---');
disp('Estimated numerator b_est_ov4 ='); disp(b_est_ov4);
disp('Estimated denominator a_est_ov4 ='); disp(a_est_ov4);
disp(['MSE of ARMA(4,4) fit: ', num2str(mse_ov4)]);

%% Over-estimating with 5-zero, 5-pole

p_ov5 = 5; 
q_ov5 = 5;
[b_est_ov5, a_est_ov5] = stmcb(h, q_ov5, p_ov5);
h_est_ov5 = impz(b_est_ov5, a_est_ov5, N);
mse_ov5 = mean((h - h_est_ov5).^2);

disp('--- Over-specified ARMA(5,5) via stmcb ---');
disp('Estimated numerator b_est_ov ='); disp(b_est_ov5);
disp('Estimated denominator a_est_ov ='); disp(a_est_ov5);
disp(['MSE of ARMA(5,5) fit: ', num2str(mse_ov5)]);

%% (C) with white Noise: y(n) = h(n) + v(n)

noiseVars = [1e-4, 1e-3, 1e-2, 1e-1];
mse_noisy = zeros(size(noiseVars));

disp(' ');
disp('--- (C) Noise sensitivity experiments ---');
for k = 1:length(noiseVars)
    sigma2 = noiseVars(k);
    v = sqrt(sigma2)*randn(size(h));  
    y = h + v;                        

    [b_est_n, a_est_n] = stmcb(y, q, p);
   
    h_est_n = impz(b_est_n, a_est_n, N);
    mse_noisy(k) = mean((h - h_est_n).^2);

    disp(['Noise var = ', num2str(sigma2), ...
          ', b_est_n = ', mat2str(b_est_n,4), ...
          ', a_est_n = ', mat2str(a_est_n,4), ...
          ', MSE vs. true h(n) = ', num2str(mse_noisy(k))]);
end

%% (D) Prony's Method

[b_prony, a_prony] = prony(h, q, p); 
h_prony = impz(b_prony, a_prony, N);
mse_prony = mean((h - h_prony).^2);

disp(' ');
disp('--- (D) Prony''s method ARMA(2,4) results ---');
disp('b_prony ='); disp(b_prony);
disp('a_prony ='); disp(a_prony);
disp(['MSE of Prony(2,4) fit: ', num2str(mse_prony)]);

%% Prony with white Noise: y(n) = h(n) + v(n)
noiseVars = [1e-4, 1e-3, 1e-2, 1e-1];
mse_noisy = zeros(size(noiseVars));

disp(' ');
disp('--- (D) Noise sensitivity experiments, Prony ---');
for k = 1:length(noiseVars)
    sigma2 = noiseVars(k);
    v = sqrt(sigma2)*randn(size(h));  
    y = h + v;       

    [b_est_n, a_est_n] = prony(y, q, p); 
   
    h_est_n = impz(b_est_n, a_est_n, N);
    mse_noisy(k) = mean((h - h_est_n).^2);

    disp(['Noise var = ', num2str(sigma2), ...
          ', b_est_n = ', mat2str(b_est_n,4), ...
          ', a_est_n = ', mat2str(a_est_n,4), ...
          ', MSE vs. true h(n) = ', num2str(mse_noisy(k))]);
end

%% Comparison plots

figure;
subplot(3,1,1)
stem(h, 'filled'); 
title('True h(n)');
xlabel('n'); ylabel('Amplitude');
grid on;

subplot(3,1,2)
stem(h_est, 'filled'); 
title('Iterative Prefiltering ARMA(2,4) estimate');
xlabel('n'); ylabel('Amplitude');
grid on;

subplot(3,1,3)
stem(h_prony, 'filled'); 
title('Prony Method ARMA(2,4) estimate');
xlabel('n'); ylabel('Amplitude');
grid on;

sgtitle('Impulse Response Comparison');

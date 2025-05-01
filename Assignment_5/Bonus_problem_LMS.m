%% Bonus Problem
clear; clc;

N = 5000; % samples
sigma2 = 0.1;              
true_h = [0.5 -1 -2 1 0.5]; 
order_list = [5, 10, 3];
mu_values = 1e-6:1e-6:1e-3; % selected step size
x = sqrt(sigma2) * randn(N, 1);

d = conv(x, true_h, 'same'); % conv to get desire

for case_num = 1:length(order_list)
    M = order_list(case_num);
    fprintf('\n Case (%c): LMS filter order = %d \n', 'a' + case_num - 1, M);
    
    for mu = mu_values
        [w, e] = lms(x, d, M, mu);
        mse = mean(e(end-100:end).^2); % neglect transient
        if mse < 0.4 % selected threshold
            fprintf('  Converged for mu = %.3f (MSE ≈ %.4f)\n', mu, mse);
            break;
        end
    end
end

%%  Adaptive LMS function
function [w, e] = lms(x, d, M, mu)

N = length(x); % get input length
w = zeros(M, 1); % weight
e = zeros(N, 1);        

for n = M:N
    x_buff = x(n:-1:n-M+1);       
    y = w' * x_buff;              
    e(n) = d(n) - y;                
    w = w + mu * x_buff * e(n);    
end
end

%% b, getting A
k = linspace(0, 1, 100);     
B = 10;                     
A = Fourier_Mat(k', B);                   

%% c, t2 moving  
B = 10;  
N = 2*B + 1; 
M = N; 

t = linspace(0, 1, M); 
t2_moving = linspace(t(2), t(1), 20); 

sigma_stored = zeros(1, length(t2_moving)); 

for i = 1:length(t2_moving)
    t_change = t; 
    t_change(2) = t2_moving(i); 
    
    A = zeros(M, N);
    A(:, 1) = 1; 

    for j = 1:B
        for k = 1:M
            A(k, j+1) = sqrt(2) * cos(2 * pi * j * t_change(k));
            A(k, B + 1 + j) = sqrt(2) * sin(2 * pi * j * t_change(k));
        end
    end

    [u, s, v] = svd(A);
    sigma_stored(i) = min(diag(s)); 
end

figure (1);
clf;
plot(t2_moving, sigma_stored, 'm-o', 'LineWidth', 1.5);
set(gca, 'XDir', 'reverse');
xlabel('t2 value changing');
ylabel('Smallest singular value of A');
title('Impact of moving t_2 closer to t1 on smallest singular value');
grid on;

%% d, estimation of x(t)
load("sampling.mat");
N = 11;
B = (N-1)/2;
t_smooth = linspace(0, 1, 500);
A_smooth = Fourier_Mat(t_smooth, B);

% Ta, ya
A_a = Fourier_Mat(Ta, B);
alpha_a = pinv(A_a) * ya;
x_a = A_smooth * alpha_a;

% Tb, yb
A_b = Fourier_Mat(Tb, B);
alpha_b = pinv(A_b) * yb;
x_b = A_smooth * alpha_b;

% Ta, ya
A_c = Fourier_Mat(Tc, B);
alpha_c = pinv(A_c) * yc;
x_c = A_smooth * alpha_c;

% Ta, ya
A_d = Fourier_Mat(Td, B);
alpha_d = pinv(A_d) * yd;
x_d = A_smooth * alpha_d;

figure (2);
clf;
plot(t_smooth, x_a, "r", "LineWidth", 1.5, "DisplayName", "Estimation with Ta, ya");
hold on;
plot(t_smooth, x_b, "c", "LineWidth", 1.5, "DisplayName", "Esitmation with Tb, yb");
hold on;
plot(t_smooth, x_c, "m", "LineWidth", 1.5, "DisplayName", "Esitmation with Tc, yc");
hold on;
plot(t_smooth, x_d, "b", "LineWidth", 1.5, "DisplayName", "Esitmation with Td, yd");
legend ();
xlabel("t");
ylabel("x(t)");
title("Estimation of x(t)");
grid on;

%% 











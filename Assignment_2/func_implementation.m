%% Problem 1
img = imread('trees.jpg'); 

%% Problem 1_a, Compression implementation
dct_matrix = DCT_calculate(img);

% save('DCT_M','dct_matrix')

%% Problem 1_a, Reconstruction implementation

DCT = dct_matrix;
idct_matrix = IDCT_calculate(DCT);

%% Problem 1_b, keeping the first M coeff
M = 1;
Mth_keep = Mth_coeff(img, M);
reduced_img = IDCT_calculate(Mth_keep);

%% Problem 1_c,
man_img = imread('man.tif');
x = double(man_img);

x_norm = norm(x, 'fro')^2;

M_coe = round(linspace(1, 64, 10)); 
normalized_E = zeros(1, length(M_coe));

for i = 1:length(M_coe)
    M = M_coe(i);
    keep_values = Mth_coeff(man_img, M);
    xm = keep_values + 128; % don't forget this
    least_sqerr = norm(x - xm, 'fro')^2;
    normalized_E(1, i) = log10(least_sqerr / x_norm);
    disp(['M = ', num2str(M), ', log Error = ', num2str(normalized_E(1, i))]);
end

figure (1);
clf;
plot(M_coe, abs(normalized_E), 'o-', 'LineWidth', 1.5);
xlabel("M (Number of DCT Coefficients Kept)");
ylabel("log_{10}(Normalized Error)");
title("DCT Compression Error vs M");
grid on;

%% Problem 1_d,
man_img = imread('man.tif');
x = double(man_img);
x_norm = norm(x, 'fro')^2;

Q = load("jpeg_Qtable.mat");
Q = Q.Q;
M_coe = round(linspace(1, 64, 10)); 
normalized_E = zeros(1, length(M_coe));

for i = 1:length(M_coe)
    M = M_coe(i);
    keep_values = Mth_Quantize(man_img, M, Q);
    xm = keep_values + 128; % don't forget this
    least_sqerr = norm(x - xm, 'fro')^2;
    normalized_E(1, i) = log10(least_sqerr / x_norm);
    disp(['M = ', num2str(M), ', log Error = ', num2str(normalized_E(1, i))]);
end

figure (2);
clf;
plot(M_coe, abs(normalized_E), 'o-', 'LineWidth', 1.5);
xlabel("M (Number of DCT Coefficients Kept with Quantization)");
ylabel("log_{10}(Normalized Error)");
title("DCT Compression Error vs M with Q");
grid on;

%% 




%% 
function A = Fourier_Mat(k, B)

    M = length(k);    
    N = 2*B + 1;
  
    A = zeros(M, N);    
    A(:, 1) = 1; % for a0

    for i = 1:B
        A(:, i+1) = sqrt(2) * cos(2 * pi * i * k);     % Cosine 
        A(:, B + 1 + i) = sqrt(2) * sin(2 * pi * i * k); % Sine
    end
end

%% 

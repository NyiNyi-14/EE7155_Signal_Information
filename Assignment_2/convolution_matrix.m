%% 
function A = convolution_matrix(h, N)
    L = length(h); % Length of filter h
    M = N + L - 1; % Length of y after convolution

    % Initialize A as M × N matrix of zeros
    A = zeros(M, N);

    % Fill in the matrix using the filter h
    for i = 1:N
        A(i:i+L-1, i) = h; % Assign h to appropriate shifted location
    end
end
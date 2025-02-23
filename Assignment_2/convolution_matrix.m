%% Problem 7_a
function A = convolution_matrix(h, x)
    L = length(h); % Length of filter h
    [N, ~] = size(x); % input row and column
    M = N + L - 1; % Length of y or A after convolution

    A = zeros(M, N);

    for i = 1:N
        A(i:i+L-1, i) = h; %  h to shifted location
    end
end

%% 
%% Problem_1(b), keeping Mth coeff

function compressed = Mth_Quantize(image, M, Q)

    if size(image, 3) == 3
        gray_img = rgb2gray(image); % ensure RGB to Grey scale
    else
        gray_img = image;
    end
    
    gray_img = double(gray_img); % change uint8 to float
    gray_img = gray_img - 128; % make zero-centered [-128,127] not [0,255]
    [row, column] = size(gray_img);
    
    row_div_8 = ceil(row/8) * 8; % ensure row divisible by 8
    col_div_8 = ceil(column/8) * 8; % ensure column divisible by
    
    fill_matrix = padarray(gray_img, [row_div_8 - row, col_div_8 - column], ...
                "replicate", "post"); % fill the missing values 
    
    compressed = zeros(size(fill_matrix));
    zigzag_idx = jpgzzind(8,8);
    
    for i = 1:8:size(fill_matrix, 1) % row
        for j = 1:8:size(fill_matrix, 2) % column
 
            bl_8x8 = fill_matrix(i:i+7, j:j+7); % get 8×8 block
    
            dct_block = dct2(bl_8x8); % 2D DCT
    
            flatten_com = reshape(dct_block.', 64, 1);
    
            low_freq_first = flatten_com(zigzag_idx);
            low_freq_first(M+1:end) = 0; % eliminate the rest

            rearrange = zeros(64,1); % back to original 8×8
            rearrange(zigzag_idx) = low_freq_first;
            rearrange = reshape(rearrange, 8, 8).';
            quantized = Q .* round(rearrange./Q);

            compressed(i:i+7, j:j+7) = quantized; % store in normal format
    
        end
    end

end

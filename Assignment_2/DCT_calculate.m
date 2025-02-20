%% Problem_1(a), DCT 8x8
function compressed = DCT_calculate(image)

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

    for i = 1:8:size(fill_matrix,1)

        for j =1:8:size(fill_matrix, 2)

            bl_8x8 = fill_matrix(i:i+7, j:j+7);
            compressed(i:i+7, j:j+7) = dct2(bl_8x8);

        end

    end

    figure;
    imshow(uint8(gray_img)), title('Original Image in Gray Scale');
%     subplot(1,2,2), imshow(uint8(image)), title('Original Image in RGB');
    disp("DCT row: " + size(compressed,1));
    disp("DCT column: " + size(compressed,2));

end

%% 


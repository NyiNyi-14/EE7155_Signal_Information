%% Problem_1(a), IDCT_reconstruction

function reconstruction = IDCT_calculate(DCT)
    reconstruction = zeros(size(DCT));

    for i = 1:8:size(reconstruction,1)

        for j = 1:8:size(reconstruction,2)

            bl_8x8 = DCT(i:i+7, j:j+7);
            reconstruction(i:i+7, j:j+7) = idct2(bl_8x8);

        end

    end

    reconstruction = reconstruction + 128; % back to [0, 255]
    figure;
    imshow(uint8(reconstruction)), title('Reconstructed Image in Gray Scale');
%     subplot(1,2,2), imshow(uint8(repmat(reconstruction, [1, 1, 3]))), title('Reconstructed Image in RGB');

end

%% 



function error_image = display_error_image(image1,image2, show_bar)
%DISPLAY_ERROR_IMAGE calculate the error image between two input image and
%display it in the heatmap form
error_image = abs(image1-image2);
if show_bar
    figure;imagesc(error_image);colorbar; caxis([0,200]); axis square; axis off;
else
    figure;imagesc(error_image); caxis([0,200]); axis square; axis off;
end


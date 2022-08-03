function RMSE = Cal_ROI_RMSE(NoiseImg, TrueImg, ROI_pt1, ROI_pt2)
    ROI_TrueImg = TrueImg(ROI_pt1(1):ROI_pt1(2),ROI_pt2(1):ROI_pt2(2));
    ROI_NoiseImg = NoiseImg(ROI_pt1(1):ROI_pt1(2),ROI_pt2(1):ROI_pt2(2));
    RMSE = norm(ROI_TrueImg - ROI_NoiseImg, 'fro') / numel(ROI_NoiseImg);
end


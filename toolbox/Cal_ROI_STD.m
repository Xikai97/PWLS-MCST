function STD = Cal_ROI_STD(Img, ROI_pt1, ROI_pt2)
    ROI_Img = Img(ROI_pt1(1):ROI_pt1(2),ROI_pt2(1):ROI_pt2(2));
    STD = std2(ROI_Img);
end
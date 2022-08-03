function [profile_set] = Get_profiles_comparison(imageset, baseline, legend_tag)
%GET_PROFILES_COMPARISON This function aims to obtain profiles of imageset
% in one certain baseline location. Afterthat, it will display a figure to
% illustrate profiles straightly.
% input: imageset is supposed to be in the form of MxNxK, where K is stand
% for the number of images
% input: baseline locates the line we need to compare and its' range is [0,1](Normalized)
% Output: profile_set(NxK) contains profile data for each input image.
[height, width, imageNum] = size(imageset);
baseline_specific = round(height*baseline);
profile_set = reshape(imageset(baseline_specific,:,:),width,imageNum);

figure;
for k = 1 : imageNum
    handles(k) = plot(profile_set(1:end,k));hold on;
    lables{k} = sprintf('cluster %d',k);
end
% legend(handles,lables{:});
legend(handles,legend_tag);
set(gca,'XGrid','on');
set(gca,'YGrid','on');
set(gca,'XTick',0:150:600);
set(gca,'YTick',0:400:1600);
end


function [ matImag ] = patch2volume( matSignal, sizeImage, sizePatch, slidingDis)
% patch2volume sticks patches back to the 3-d image volume.
% matSignal: the matrix containing vertical signal vectors
% sizeImage: [rows cols thickness] 1x3 vector
% nSlideDist:[row col thickness] 1x3 vector sliding distance
% Not wrap-around

% The image to recover
matImag = zeros(sizeImage,'double');
% The average weight
% matAvgWeight = zeros(sizeImage,'single');
% mark the places where a patch should be inserted by 1
idxMat = zeros(sizeImage-sizePatch+1,'double');
idxMat([[1:slidingDis(1):end-1],end],[[1:slidingDis(2):end-1],end],[[1:slidingDis(3):end-1],end]) = 1;
% record the positions of each marked entry in idxMat
idx = find(idxMat);
[rows, cols, deps] = ind2sub(size(idxMat), idx);
% rows = single(rows);cols = single(cols);deps = single(deps);
for ii = 1:length(idx)
    curBlock = reshape(matSignal(:, ii), sizePatch);
    matImag(rows(ii):rows(ii)+sizePatch(1)-1,cols(ii):cols(ii)+sizePatch(2)-1,deps(ii):deps(ii)+sizePatch(3)-1) ...
  = matImag(rows(ii):rows(ii)+sizePatch(1)-1,cols(ii):cols(ii)+sizePatch(2)-1,deps(ii):deps(ii)+sizePatch(3)-1) + curBlock;

%     matAvgWeight(rows(ii):rows(ii)+sizePatch(1)-1,cols(ii):cols(ii)+sizePatch(2)-1,deps(ii):deps(ii)+sizePatch(3)-1) ...
%   = matAvgWeight(rows(ii):rows(ii)+sizePatch(1)-1,cols(ii):cols(ii)+sizePatch(2)-1,deps(ii):deps(ii)+sizePatch(3)-1) + 1;
end
% matImag = matImag ./ matAvgWeight;

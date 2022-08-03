function [blocks,idx] = volume2patch(matImag,sizePatch,slidingDis) 
% I: 3D image volume
% blkSize: 1x3 vector [row column thickness]
% slidingDis: 1x3 vector [row column thickness] 
idxMat = zeros(size(matImag)-sizePatch+1,'double');
idxMat([[1:slidingDis(1):end-1],end],[[1:slidingDis(2):end-1],end],[[1:slidingDis(3):end-1],end]) = 1;
idx = find(idxMat);
[rows, cols, deps] = ind2sub(size(idxMat),idx); 
% rows = single(rows);cols = single(cols);deps = single(deps);
blocks = zeros(prod(sizePatch),length(idx),'double');
for i = 1:length(idx)
    currBlock = matImag(rows(i):rows(i)+sizePatch(1)-1,cols(i):cols(i)+sizePatch(2)-1, deps(i):deps(i)+sizePatch(3)-1);
    blocks(:,i) = currBlock(:);
end

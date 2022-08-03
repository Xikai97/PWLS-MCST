function [x_recover] = recover_downsampling(x_down,down_idx)
    x_recover = zeros(size(down_idx,1),size(x_down,2));
    iter = 1;
    for i=1:1:size(down_idx,1)
        if down_idx(i,1)==1
            x_recover(i,:) = x_down(iter,:);
            iter = iter + 1;
        end
    end
end
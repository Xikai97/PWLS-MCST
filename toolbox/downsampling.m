function [x_down,down_idx] = downsampling(x,down_size)
% used for pooling strategy in MRST model
    original_size = size(x,1);
    x_norm = sum(x.^2,2);
    [~,Order] = sort(x_norm);
    down_idx = ones(original_size,1);
    down_idx(Order(1:original_size-down_size),1) = 0;
    Order_inverse = sort(Order(original_size-down_size+1:original_size));
    x_down = x(Order_inverse,:);
end
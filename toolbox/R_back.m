function x = R_back(arg, y)
%%% perform adjoint transform

% input: matrix_{dim of transform x # of patches}
% output: 3d image 
% x = patch2volume( arg.transf' * y, arg.ImgSize, arg.PatSize, arg.SldDist);
x = col2imstep( single(arg.transf' * y), arg.ImgSize, arg.PatSize, arg.SldDist);
end
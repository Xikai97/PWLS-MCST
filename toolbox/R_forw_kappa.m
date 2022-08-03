function y = R_forw_kappa(arg, x)
%%% perform forward transform

% input: 3d image
x = x .* arg.mask;
% output: [C P_1 x ... C P_{# of patches} x] 

y = arg.transf * im2colstep(single(arg.kappa.*x), arg.PatSize, arg.SldDist);

end
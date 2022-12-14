%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xuehang Zheng, UM-SJTU Joint Institute
clear; close all; 
addpath(genpath('./data_XCAT'));
addpath(genpath('../toolbox'));
%% generate noisy sinogram and statistical weighting
I0 = 1e4;

down = 1; % downsample rate
sg = sino_geom('ge1', 'units', 'mm', 'strip_width', 'd', 'down', down);
% sg.na = 123; % sparse view

dir = ['./data_XCAT/slice20/' num2str(I0)];

load('./data_XCAT/slice20/slice#20_840.mat');  %slice with 840x840 is applied here


mm2HU = 1000 / 0.02;
% ig_big = image_geom('nx', 420, 'dx', 500/1024);
ig_big = image_geom('nx', 840, 'dx', 500/1024);
% ig_big = image_geom('nx',1024, 'dx', 500/1024);

Abig = Gtomo_nufft_new(sg, ig_big);  clear ig_big;
sino_true = Abig * xtrue_hi; clear Abig xtrue_hi;

fprintf('adding noise...\n');
yi = poisson(I0 * exp(-sino_true ./mm2HU), 0, 'factor', 0.4);
var = 5;
ye = var.* randn(size(yi)); % Gaussian white noise ~ N(0,std^2)
k = 1;
zi = k * yi + ye; clear yi ye;
error = 1/1e1 ;
zi = max(zi, error);   
sino = -log(zi ./(k*I0)) * mm2HU; 

wi = (zi.^2)./(k*zi + var^2);  % updated on 2017-05-11 
save([dir '/wi.mat'], 'wi');    
save([dir '/sino_fan.mat'], 'sino'); 
% figure name 'Noisy sinogram'
% imshow(sino, [2 40000]);

%% setup target geometry and fbp
ig = image_geom('nx', 420, 'dx', 500/512, 'down', down);
% ig = image_geom('nx', 512, 'dx', 500/512, 'down', down);

ig.mask = ig.circ > 0;
A = Gtomo_nufft_new(sg, ig);
fprintf('fbp...\n');
tmp = fbp2(sg, ig);
xfbp = fbp2(sino, tmp, 'window', 'hanning,0.4'); clear tmp;
xfbp = max(xfbp, 0);
save([dir '/xfbp.mat'], 'xfbp');
figure name 'xfbp'
imshow(xfbp, [800 1200]);colorbar

%% setup kappa
fprintf('calculating kappa...\n');
kappa = sqrt( div0(A' * wi, A' * ones(size(wi))) );
save([dir '/kappa.mat'], 'kappa');

%% setup diag{A'WA1}
printm('Pre-calculating denominator D_A...');
denom = A' * col(reshape(sum(A'), size(wi)) .* wi); 
save([dir '/denom.mat'], 'denom');

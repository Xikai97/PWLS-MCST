%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xikai Yang, UM-SJTU Joint Institute
clear ; close all;
addpath(genpath('./data_Mayo'));
addpath(genpath('../toolbox'));
%% setup target geometry and weight
down = 1; % downsample rate
% sg = sino_geom('ge1', 'units', 'mm', 'strip_width', 'd', 'down', down);
sg = sino_geom('fan', 'units', 'mm', 'nb',736, 'na',1152,'orbit',360, 'ds',1.2858,...
     'strip_width','ds','dsd',1085.6,'dso',595,'dfs',0, 'down', down);  % by Siqi
% sg.na = 123;
% ig = image_geom('nx', 512, 'dx', 500/512, 'down', down);
ig = image_geom('nx',512,'fov',sg.rfov*sqrt(2)); % dx=0.69298mm, fov=354.8065mm  % by Siqi
% ig.mask = ig.circ > 0;
% A = Gtomo2_dscmex(sg, ig,'nthread', maxNumCompThreads*2);
% A = Gtomo2_dscmex(sg, ig,'nthread', jf('ncore')*2-1);
Ab = Gtomo_nufft_new(sg, ig);

% if neccessary, one could modify maxNumComThreads/jf('ncore') to make full
% use of threads of your machine to accelerate the computation of
% forward and back projections.

%% load external parameter
I0 = 1e4; % photon intensity
% load PWLS-EP Recon as initialization: 
load('./Init_EP/3mm_L067_slice120_1e4_4blk_15.5_I2b_100iter_2Dreg_20delta.mat'); % change intial EP image when Recon slice and I0 are changed!
xrlalm = info.xrlalm;

%load transform: mOmega 
load('./learned_trans_mcst_mayo/mayo_mcst_nodropmean_5_5_numBlock_80_60_eta_1000iter_8PatchSize.mat');
numBlock = info(1).numBlock;
layerNum = size(numBlock,2);
mOmega = cell(1,layerNum);
for l = 1:1:layerNum
    mOmega{l} = info(l).mOmega;
end

clear info

%load ground truth image: xtrue
load('./data_Mayo/L067/L067_slice120_3mm.mat');
xtrue = L067_slice120_3mm; % Make it consistent with Recon slice!

%load external sinogram, weight, fbp...
dir = ['./data_Mayo/L067/L067_slice120_3mm/' num2str(I0)];

printm('Loading external sinogram, weight, fbp...');
load([dir '/sino_fan.mat']);
load([dir '/wi.mat']);
load([dir '/xfbp.mat']);
load([dir '/kappa.mat']);
% figure name 'xfbp'
% imshow(xfbp, [800 1200]);

%% setup edge-preserving regularizer
ImgSiz =  [ig.nx ig.ny];  % image size
PatchSize = 8;
PatSiz = PatchSize * [1 1];  % patch size
SldDist = [1 1];         % sliding distance

nblock = 1;            % Subset Number
nIter = 2;             % I--Inner Iteration
nOuterIter = 1500;     % T--Outer Iteration
CluInt = 1;            % Clustering Interval
isCluMap = 0;          % The flag of caculating cluster mapping
pixmax = inf;          % Set upper bond for pixel values
printm('Pre-calculating denominator D_A...');
load([dir '/denom.mat']);

% setup parameters [beta,gamma_1,gamma_2,...,gamma_L]
para_set = [1.8e4,30,10;
           ];
for index = 1:size(para_set,1)
    beta = para_set(index,1);
    gamma = para_set(index,2:end);

% pre-compute D_R
PP = im2colstep(ones(ImgSiz,'single'), PatSiz, SldDist);
PatNum = size(PP, 2);
KK = col2imstep(single(PP), ImgSiz, PatSiz, SldDist);
D_R = 2 * layerNum * beta * KK(ig.mask); clear PP KK
R = Reg_MRULTRAL(ig.mask, ImgSiz, PatSiz, PatNum, SldDist, beta, gamma, mOmega, numBlock, CluInt);

info = struct('intensity',I0,'ImgSiz',ImgSiz,'SldDist',SldDist,'beta',beta,'gamma',gamma,'nblock',nblock,'nIter',nIter,'CluInt',CluInt,'pixmax',pixmax,...
        'transform',[],'xrla',[],'R_image',[],'vIdx',[],'ClusterMap',[],'RMSE',[],'SSIM',[],'PSNR',[],'relE',[],'perc',[],'idx_change_perc',[],'cost',[]);

xini = xrlalm .* ig.mask;    %initial EP image
xrla_msk = xrlalm(ig.mask);
% xini = xfbp .* ig.mask;     %initial FBP image
% xrla_msk = xfbp(ig.mask);   clear xfbp
info.xrla = xini;

%% Recon
SqrtPixNum = sqrt(sum(ig.mask(:)>0)); % sqrt(pixel numbers in the mask)
stop_diff_tol = 1e-3; % HU
% iterate_fig = figure(55);
idx_old = ones([layerNum,PatNum],'single');
info.vIdx = idx_old;
info.perc = cell(1,layerNum);
info.ClusterMap = zeros([ImgSiz,layerNum]);
for ii=1:nOuterIter
    xold = xrla_msk;
    AAA(1,ii) = norm(xrla_msk - xtrue(ig.mask)) / SqrtPixNum;
    fprintf('RMSE = %g\n', AAA(1,ii));
    info.RMSE = AAA(1,:);
    AAA(2,ii)= ssim(info.xrla, xtrue);
    fprintf('SSIM = %g\n', AAA(2,ii));
    info.SSIM = AAA(2,:);
    
    fprintf('Iteration = %d:\n', ii);
    [xrla_msk, cost] = pwls_os_rlalm(xrla_msk, Ab, reshaper(sino, '2d'),  reshaper(wi, '2d'),...
        R, denom, D_R, 'pixmax', pixmax, 'chat', 1, 'alpha', 1.999, 'rho', [],'niter', nIter);
    
    info.cost(:,ii) = cost;
    fprintf('Sum Cost = %g\n', info.cost(1,ii));
    info.R_image = R.get_spar_image();
    
    [tmp_perc,tmp_vIdx] = R.nextOuterIter();
    for l = 1:1:layerNum
        fprintf('perc%g = %g\n', l,tmp_perc{l}(end));
        info.perc{l}(ii,:) = tmp_perc{l};
        info.vIdx(l,:) = tmp_vIdx{l};
    end

    info.idx_change_perc(:,ii) = nnz(idx_old - info.vIdx)/(layerNum*PatNum);
    fprintf('Idx Change Perc = %g\n', info.idx_change_perc(:,ii));
    idx_old = info.vIdx;
    
    info.relE(:,ii) =  norm(xrla_msk - xold) / SqrtPixNum;
    fprintf('relE = %g\n', info.relE(:,ii));
    if info.relE(:,ii) < stop_diff_tol
        break
    end
    
    info.xrla = ig.embed(xrla_msk);
    info.PSNR(ii) = my_psnr2(info.xrla, xtrue, max(max(xtrue)), ig);
    figure(120), imshow(info.xrla, [800 1200]); drawnow;
    
    for l = 1:1:layerNum
        % figure;
        % imagesc(info.R_image(:,:,l)); axis square; colorbar;
        info.ClusterMap(:,:,l) = ClusterMap(ImgSiz, PatSiz, SldDist, double(info.vIdx(l,:)), PatNum, numBlock(l));
    end
    
    fig_tag_init = 40;
    if mod(ii,10)==1
        if ii>1
            for l = 1:1:layerNum
                close(fig_tag_init*10+l*10);
            end
        end
        
        for l = 1:1:layerNum
            figure(fig_tag_init*10+l*10);
            for k = 1 : numBlock(l)
                handles_l(k) = plot(info.perc{l}(1:ii,k));hold on;
                lables_l{k} = sprintf('cluster %d',k);
            end
            legend(handles_l,lables_l{:});
            xlabel('Number of Iteration','fontsize',18)
            ylabel('Layer Sparity ( % )','fontsize',18)
            title(sprintf('Sparsity Level of Layer %g',l));
        end   
    end
    
end

info.transform = mOmega;

% Revise this line when Recon slice or Model depth or learned transforms are changed
save(sprintf('./recon_mcst_mayo/MayoRecon_L067_slice120_3mm_%.1e_%g_%gblk_%.1ebeta_%g_%ggam_learn80_60_iter%g_nodropmean_%gPatchSize.mat', I0, ...
    numBlock(1), numBlock(2), beta, gamma(1), gamma(2), nOuterIter, PatchSize), 'info')

figure name 'RMSE'
plot(info.RMSE,'-+')
xlabel('Number of Outer Iteration','fontsize',18)
ylabel('RMSE(HU)','fontsize',18)
legend('PWLS-MCST')

end




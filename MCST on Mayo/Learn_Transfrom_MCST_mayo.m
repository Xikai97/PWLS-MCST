%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Xikai Yang, UM-SJTU Joint Institute
clear ; close all;
addpath(genpath('../toolbox'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%  Initialization  %%%%%%%%%%%%%%%%%%%%%%%%%%%%
LayerNum = 2;
numBlock = [5,1];   % number of clusters in each layer [K_1,K_2,...,K_L]
PatchSize = 8;
PatSiz = PatchSize * [1 1];   % patch size
l = prod(PatSiz);  % patch   size
stride = 1;   % overlapping stride   
SldDist = 1 * [1 1];  % sliding distance
iter = 1000;          % iteration
inner_loop = 1;     % inner loop


% load training Mayo data
addpath(genpath('./data_Mayo'));
load('./data_Mayo/L067/L067_slice150_3mm.mat');
load('./data_Mayo/L067/L067_slice210_3mm.mat');
load('./data_Mayo/L096/L096_slice170_3mm.mat');
load('./data_Mayo/L096/L096_slice251_3mm.mat');
load('./data_Mayo/L096/L096_slice291_3mm.mat');
load('./data_Mayo/L096/L096_slice330_3mm.mat');
load('./data_Mayo/L143/L143_slice15_3mm.mat');
%% 

Train_MayoData = zeros(size(L067_slice150_3mm,1),size(L067_slice150_3mm,2),7);
Train_MayoData(:,:,1) = L067_slice150_3mm;
Train_MayoData(:,:,2) = L067_slice210_3mm;
Train_MayoData(:,:,3) = L096_slice170_3mm;
Train_MayoData(:,:,4) = L096_slice251_3mm;
Train_MayoData(:,:,5) = L096_slice291_3mm;
Train_MayoData(:,:,6) = L096_slice330_3mm;
Train_MayoData(:,:,7) = L143_slice15_3mm;

patch=[];
for ii=1:1:7  % training data
%      for ii = 48     % testing data
    image = Train_MayoData(:, :, ii);
    % The Mathworks 'im2col' is quicker but only for stride 1.
    %    patch_tmp = im2col(image, sqrt(l) * [1 1], 'sliding');
    %     [patch_tmp, ~] = image2patch(image,sqrt(l) * [1 1], stride);
    patch_tmp = im2colstep(single(image), sqrt(l)*[1 1], stride*[1 1]);
    patch = [patch patch_tmp];
end
patch = double(patch);
% patch = patch - mean(patch);   %drop patch mean
PatNum = size(patch, 2);
ImgSiz = size(image);
fprintf('Length of training set: %d\n', PatNum);

eta_set = [80,60;
          ];
for index = 1:size(eta_set,1)
    eta = eta_set(index,:);

% initialize cluster index, transforms, sparsecodes, residual maps
IDX = cell(1,LayerNum);
[IDX{1}, ~] = kmeans(patch',numBlock(1));   % K-mean Initialization
for i = 2:1:LayerNum
    IDX{i} = randi(numBlock(i),PatNum,1);    % Random Initialization
end

TransWidth = prod(PatSiz);
D = kron(dctmtx(PatSiz(1)),dctmtx(PatSiz(2))); % DCT Initialization
mOmega = cell(1,LayerNum);
mOmega{1} = zeros(TransWidth,TransWidth,numBlock(1), 'double');
for j = 1:numBlock(1)
    mOmega{1}(:,:,j) = D;
    % mOmega{1}(:,:,j) = rand(TransWidth,TransWidth);
end
for i = 2:1:LayerNum
    mOmega{i} = zeros(TransWidth,TransWidth,numBlock(i), 'double');
    for j = 1:1:numBlock(i)
        % mOmega{i}(:,:,j) = eye(size(D));
        mOmega{i}(:,:,j) = rand(TransWidth,TransWidth);
    end
end

z = cell(1,LayerNum);
R = cell(1,LayerNum+1);
for i = 1:1:LayerNum
    z{i} = zeros(size(patch));
    R{i+1} = zeros(size(patch));
end
R{1} = patch;

% allocate space for sparsity percentage function and cost function
perc = cell(1,LayerNum);
for i = 1:1:LayerNum
    perc{i} = zeros(iter,numBlock(i)+1,'single');  % sparsity (percentage)
end
cost = zeros(iter,LayerNum*2);
% rmse = zeros(iter,LayerNum*2);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Training %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
for j = 1 : iter
    fprintf('iteration = %d:\n', j);
    % calculate backpropagation matrix
    B_matrix = cal_backpropagation_matrix(mOmega, IDX, z);
    
    for l = 1:1:LayerNum
        
        accumlate_Bmatrix = zeros(size(patch));
        if l < LayerNum
            for i = l+1:LayerNum
                accumlate_Bmatrix = accumlate_Bmatrix + B_matrix{l,i};
            end
        end

        % layer l clustering
        errorl = zeros(numBlock(l), PatNum, 'double');
        R_clu = cell(1,LayerNum+1);
        R_clu(1:l) = R(1:l);
        for k = 1:1:numBlock(l)
            zl_clu = mOmega{l}(:,:,k)*R_clu{l} - (1/(LayerNum-l+1))*accumlate_Bmatrix; % ML-test-v1  
            % zl_clu = mOmega{l}(:,:,k)*R_clu{l};  % ML-test-v2
            zl_clu = zl_clu.*(abs(zl_clu)>eta(l)/sqrt(LayerNum-l+1));
            % update R_clu since zl_clu changes
            R_clu{l+1} = mOmega{l}(:,:,k)*R_clu{l} - zl_clu;
            errorl(k, :) = sum(R_clu{l+1}.^2,'double') + eta(l)^2 * sum(abs(zl_clu) > 0);
            for r = l+2:1:LayerNum+1
                R_clu{r} = cluster_product(IDX{r-1}, mOmega{r-1}, R_clu{r-1}, 0);
                R_clu{r} = R_clu{r} - z{r-1};
                errorl(k, :) = errorl(k, :) + sum(R_clu{r}.^2,'double'); 
            end
        end
        %%%%%%%%% clustering %%%%%%%%%%%%%%
        [~, IDX{l}] = min(errorl, [] ,1);
        clear  errorl zl_clu R_clu
        
        for inner_iter = 1:1:inner_loop
            
        fprintf('Update for layer %d, inner loop = %d:\n', l,inner_iter);

        % update sparsecode zl
        OmgR_l = cluster_product(IDX{l}, mOmega{l}, R{l}, 0);
        z{l} = OmgR_l - (1/(LayerNum-l+1))*accumlate_Bmatrix;
        z{l} = z{l}.*(abs(z{l})>eta(l)/sqrt(LayerNum-l+1));
        R{l+1} = OmgR_l - z{l};
        
        % check cost function and sparsity level
        cost(j,2*l-1) = sum(sum(abs(z{l}) > 0));
        cost(j,2*l) = sum(sum(R{l+1}.^2,'double'));
        perc{l}(j,:) = get_cluster_sparsity(IDX{l}, z{l}, numBlock(l));
        fprintf('layer%g sparsity = %g\n', l, perc{l}(j,numBlock(l)+1));
        
        % update transform mOmegal
        for k = 1:1:numBlock(l)
            tmp_idx = IDX{l}==k;
            if (size(R{l}(:,tmp_idx),2) > 0)
                [Ul_k,~,Vl_k]=svd(R{l}(:,tmp_idx)*(z{l}(:,tmp_idx) + (1/(LayerNum-l+1))*accumlate_Bmatrix(:,tmp_idx))');
                mOmega{l}(:,:,k) = Vl_k*Ul_k'; 
            else
                fprintf('R%g cluster %g is empty\n', l,k);
            end
        end
        
        end

    end
    
    %%%% transform visualization %%%%%%%%%
    fig_tag_init = 40;
    for l = 1:1:LayerNum
        figure(fig_tag_init+l*10);
        for k = 1:1:numBlock(l)
            subplot(1,numBlock(l),k);display_transform_changeSize(mOmega{l}(:,:,k), PatchSize);
        end
        title(sprintf('Learned Transform of Layer %g',l));
    end
    
    %%%% check cluster-mapping and residual-mapping %%%%%%%%
    CluMap = cell(1,LayerNum);
    R_image = cell(1,LayerNum);
    for l = 1:1:LayerNum
        CluMap{l} = ClusterMap(ImgSiz, PatSiz, SldDist, IDX{l}(:,end*6/7+1:end), PatNum, numBlock(l));
        R_image{l} = col2imstep(single(R{l+1}(:,size(patch,2)*6/7+1:end)),size(image),PatSiz)/prod(PatSiz);
    end
  
    
    if mod(j,10)==1
        if j>1
            for l = 1:1:LayerNum
                close(fig_tag_init*10+l*10);
            end
        end
        
        for l = 1:1:LayerNum
            figure(fig_tag_init*10+l*10);
            for k = 1 : numBlock(l)
                handles_l(k) = plot(perc{l}(1:j,k));hold on;
                lables_l{k} = sprintf('cluster %d',k);
            end
            legend(handles_l,lables_l{:});
            xlabel('Number of Iteration','fontsize',18)
            ylabel('Layer Sparity ( % )','fontsize',18)
            title(sprintf('Sparsity Level of Layer %g',l));
        end   
    end
    
end

info = struct('mOmega',mOmega,'eta',eta,'ImgSiz',size(image),'SldDist',SldDist,...
    'numBlock',numBlock,'iter',iter,'IDX',IDX,'perc',perc,'CluMap',CluMap,'cost',cost,'R_image',R_image);

save(sprintf('./learned_trans_mcst_mayo/mayo_mcst_nodropmean_%g_%g_numBlock_%g_%g_eta_%giter_%gPatchSize.mat',...
    numBlock(1),numBlock(2),eta(1),eta(2),iter,PatchSize), 'info')
end


function B_matrix = cal_backpropagation_matrix(mOmega, IDX, z)
    LayerNum = size(mOmega,2);
    B_matrix = cell(LayerNum-1,LayerNum);
    % generate each cell row by row
    for rowIdx = 1:1:LayerNum-1
       B_matrix{rowIdx,rowIdx+1} = cluster_product(IDX{rowIdx+1}, mOmega{rowIdx+1}, z{rowIdx+1}, 1);
       if rowIdx+2 <= LayerNum
          for colIdx = rowIdx+2:1:LayerNum
              B_matrix{rowIdx,colIdx} = B_matrix{rowIdx,colIdx-1} + cal_increment(mOmega(rowIdx+1:colIdx), IDX(rowIdx+1:colIdx), z(colIdx));
          end
       end
    end 
end

function incre = cal_increment(mOmega, IDX, z_last)
    tmp_LayerNum = size(mOmega,2);
    mid_term = cluster_product(IDX{end}, mOmega{end}, z_last{end}, 1);
    for i = 1:1:tmp_LayerNum-1
        mid_term = cluster_product(IDX{end-i}, mOmega{end-i}, mid_term, 1);
    end
    incre = mid_term;
end

function perc_clu = get_cluster_sparsity(cluIdx, sparsecode, cluNum)
    perc_clu = zeros(1,cluNum+1);
    for k = 1:cluNum
        sparsecode_tmp = sparsecode(:,cluIdx==k);
        perc_clu(1,k) = nnz(sparsecode_tmp)/ numel(sparsecode_tmp)* 100; 
    end
    perc_clu(1,cluNum+1) = nnz(sparsecode)/ numel(sparsecode)* 100; 
end

function product = cluster_product(cluIdx,mOmega,multiplicand,transpose_flag)
    product = zeros(size(multiplicand));
    cluNum = size(mOmega,3);
    if (transpose_flag==1)
        for k = 1:cluNum
            product(:,cluIdx==k) = mOmega(:,:,k)' * multiplicand(:,cluIdx==k);
        end
    else
        for k = 1:cluNum
            product(:,cluIdx==k) = mOmega(:,:,k) * multiplicand(:,cluIdx==k);
        end
    end
end

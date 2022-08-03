classdef Reg_MRULTRAL < handle
    
    properties
        mMask;    % the mask matrix
        ImgSiz;   % image size
        PatSiz;   % patch size
        SldDist;  % sliding distance
        layerNum; % number of layers in this model
        beta;       % trade-off para between data-fidelity and regularizer terms
        gamma;      % threshold for all layers
        mOmega;    % transform matrix for all layers
        numBlock;  % cluster number for all layers
        vIdx;     % the patch index for all layers
        isSpa;    % the flag of sparse code update
        isClu;    % the flag of all layer clustering update
        CluInt;    % the number of clustering interval
        mSpa;     % the matrix of sparse code for all layers
        R;        % the sparse error for all layers
        cost;      % contain six parts' loss of MRULTRAL
    end
    
    methods
        function obj = Reg_MRULTRAL(mask, ImgSiz, PatSiz, numPatch, SldDist, beta, gamma, mOmega, numBlock, CluInt)
            obj.layerNum = size(numBlock,2);
            obj.mMask = mask;
            obj.PatSiz = PatSiz;
            obj.ImgSiz = ImgSiz;
            obj.SldDist = SldDist;
            obj.beta = beta;
            obj.gamma = gamma;
            obj.mOmega = mOmega;
            obj.numBlock = numBlock;
            obj.isSpa = true;
            obj.isClu = CluInt * ones(1,obj.layerNum);
            obj.CluInt = CluInt;
            obj.vIdx = cell(1,obj.layerNum);
            for l = 1:1:obj.layerNum
                obj.vIdx{l} = randi(numBlock(l),numPatch,1);    % Random Initialization
            end
            obj.mSpa = cell(1,obj.layerNum);
            obj.R = cell(1,obj.layerNum+1);
            for l = 1:1:obj.layerNum
                obj.mSpa{l} = zeros(prod(PatSiz),numPatch);
                R{l+1} = zeros(prod(PatSiz),numPatch);
            end
            obj.cost = zeros(1,2*obj.layerNum);
        end
        
        function cost = penal(obj, A, x, wi, sino)
            % data fidelity
            df = .5 * sum(col(wi) .* (A * x - col(sino)).^2, 'double');
            cost_val = df;
            for l = 1:1:obj.layerNum
                cost_val = cost_val + obj.beta*(obj.cost(2*l)+obj.gamma(l)^2*obj.cost(2*l-1));
            end
            cost=[cost_val obj.cost]; 
        end
        
        function grad = cgrad(obj, x)
            x = embed(x, obj.mMask);
            patch = im2colstep(single(x), obj.PatSiz, obj.SldDist);  clear x;
            PatNum = size(patch, 2);
            obj.R{1} = patch;
            
            % calculate backpropagation matrix
            B_matrix = obj.cal_backpropagation_matrix(obj.mOmega, obj.vIdx, obj.mSpa);
            
            for l = 1:1:obj.layerNum
                if(obj.isClu(l) == obj.CluInt)
                % layer l clustering
                errorl = zeros(obj.numBlock(l), PatNum, 'double');
                R_clu = cell(1,obj.layerNum+1);
                R_clu(1:l) = obj.R(1:l);
                for k = 1:1:obj.numBlock(l)
                    % zl_clu = mOmega{l}(:,:,k)*R_clu{l} - (1/(LayerNum-l+1))*accumlate_Bmatrix;
                    zl_clu = obj.mOmega{l}(:,:,k)*R_clu{l};
                    zl_clu = zl_clu.*(abs(zl_clu)>obj.gamma(l)/sqrt(obj.layerNum-l+1));
                    % update R_clu since zl_clu changes
                    R_clu{l+1} = obj.mOmega{l}(:,:,k)*R_clu{l} - zl_clu;
                    errorl(k, :) = sum(R_clu{l+1}.^2,'double') + obj.gamma(l)^2 * sum(abs(zl_clu) > 0);
                    for r = l+2:1:obj.layerNum+1
                        R_clu{r} = obj.cluster_product(obj.vIdx{r-1}, obj.mOmega{r-1}, R_clu{r-1}, 0);
                        R_clu{r} = R_clu{r} - obj.mSpa{r-1};
                        errorl(k, :) = errorl(k, :) + sum(R_clu{r}.^2,'double'); 
                    end
                end
                %%%%%%%%% clustering %%%%%%%%%%%%%%
                [~, obj.vIdx{l}] = min(errorl, [] ,1);
                clear  errorl zl_clu R_clu
                obj.isClu(l) = 0; % reset clustering counter
                end
                %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
                % update sparsecode zl
                if l==obj.layerNum
                    diff = zeros(size(patch));
                end
                if (obj.isSpa)
                    OmgR_l = obj.cluster_product(obj.vIdx{l}, obj.mOmega{l}, obj.R{l}, 0);
                    accumlate_Bmatrix = zeros(size(patch));
                    if l < obj.layerNum
                        for i = l+1:obj.layerNum
                            accumlate_Bmatrix = accumlate_Bmatrix + B_matrix{l,i};
                        end
                    end
                    obj.mSpa{l} = OmgR_l - (1/(obj.layerNum-l+1))*accumlate_Bmatrix;
                    obj.mSpa{l} = obj.mSpa{l}.*(abs(obj.mSpa{l})>obj.gamma(l)/sqrt(obj.layerNum-l+1));
                    obj.R{l+1} = OmgR_l - obj.mSpa{l};
                    % check cost function and sparsity level
                    obj.cost(2*l-1) = sum(sum(abs(obj.mSpa{l}) > 0));
                    obj.cost(2*l) = sum(sum(obj.R{l+1}.^2,'double'));
                    if l==obj.layerNum
                        B_0k = obj.cal_B_0k(obj.mOmega, obj.vIdx, obj.mSpa);
                        accumulate_B_0k = B_0k{1};
                        for j = 2:1:obj.layerNum
                            accumulate_B_0k = accumulate_B_0k + B_0k{j};
                        end
                        diff = obj.layerNum * patch - accumulate_B_0k;
                        obj.isSpa = false;  % close the flag of sparse code update
                    end
                else
                    if l==obj.layerNum
                        B_0k = obj.cal_B_0k(obj.mOmega, obj.vIdx, obj.mSpa);
                        accumulate_B_0k = B_0k{1};
                        for j = 2:1:obj.layerNum
                            accumulate_B_0k = accumulate_B_0k + B_0k{j};
                        end
                        diff = obj.layerNum * patch - accumulate_B_0k;
                    end
                end
            end
          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
          % get gradient of Regular term
          grad = 2 * obj.beta .* col2imstep(single(diff), obj.ImgSiz, obj.PatSiz, obj.SldDist);
          grad = grad(obj.mMask);
             
       end
        
        
        function [perc,vIdx] = nextOuterIter(obj)
            obj.isClu = obj.isClu + ones(size(obj.isClu));
            vIdx = obj.vIdx;
            obj.isSpa = true; % open the flag of updating sparse code
            % sparsity check
            perc = cell(1,obj.layerNum);
            for l = 1:1:obj.layerNum
                perc{l} = obj.get_cluster_sparsity(obj.vIdx{l}, obj.mSpa{l}, obj.numBlock(l));
            end
        end
        
        function perc_clu = get_cluster_sparsity(obj, cluIdx, sparsecode, cluNum)
            perc_clu = zeros(1,cluNum+1);
            for k = 1:cluNum
                sparsecode_tmp = sparsecode(:,cluIdx==k);
                perc_clu(1,k) = nnz(sparsecode_tmp)/ numel(sparsecode_tmp)* 100; 
            end
            perc_clu(1,cluNum+1) = nnz(sparsecode)/ numel(sparsecode)* 100; 
        end
        
        function spar_image = get_spar_image(obj)
            spar_image = zeros([obj.ImgSiz,obj.layerNum]);
            for l = 1:1:obj.layerNum
                spar_image(:,:,l) = col2imstep(single(obj.R{l}), obj.ImgSiz, obj.PatSiz, obj.SldDist);
            end
        end
        
        function B_0k = cal_B_0k(obj, mOmega, IDX, z)
            LayerNum = size(mOmega,2);
            B_0k = cell(1,LayerNum);
            B_0k{1} = obj.cluster_product(IDX{1}, mOmega{1}, z{1}, 1);
            for j = 2:1:LayerNum
                B_0k{j} = B_0k{j-1} + obj.cal_increment(mOmega(1:j), IDX(1:j), z(j));
            end
        end
        
        function B_matrix = cal_backpropagation_matrix(obj, mOmega, IDX, z)
            LayerNum = size(mOmega,2);
            B_matrix = cell(LayerNum-1,LayerNum);
            % generate each cell row by row
            for rowIdx = 1:1:LayerNum-1
            B_matrix{rowIdx,rowIdx+1} = obj.cluster_product(IDX{rowIdx+1}, mOmega{rowIdx+1}, z{rowIdx+1}, 1);
            if rowIdx+2 <= LayerNum
                for colIdx = rowIdx+2:1:LayerNum
                    B_matrix{rowIdx,colIdx} = B_matrix{rowIdx,colIdx-1} + obj.cal_increment(mOmega(rowIdx+1:colIdx), IDX(rowIdx+1:colIdx), z(colIdx));
                end
            end
            end 
        end
        
        function incre = cal_increment(obj, mOmega, IDX, z_last)
            tmp_LayerNum = size(mOmega,2);
            mid_term = obj.cluster_product(IDX{end}, mOmega{end}, z_last{end}, 1);
            for i = 1:1:tmp_LayerNum-1
                mid_term = obj.cluster_product(IDX{end-i}, mOmega{end-i}, mid_term, 1);
            end
            incre = mid_term;
        end

        function product = cluster_product(obj, cluIdx, mOmega, multiplicand, transpose_flag)
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
        
        
    end
    
end


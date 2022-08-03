function ClusterMap = CalClusterMap(ImgSiz, PatSiz, SldDist, vIdx, PatNum, numBlock)
%   ImgSiz - image size   
%   PatSiz - patch size
%   SldDist - sliding distance
%   vIdx    -  indices of image patches
%   PatNum  -  the number of image patches
%   numBlock - the number of square transforms
%   ClusterMap = cluster map for every voxel
%
%   Xuehang Zheng, UM-SJTU joint institute

  idxMat = zeros(ImgSiz - PatSiz + 1, 'single');
  if length(ImgSiz) == 3   % 3D
     idxMat([[1:SldDist(1):end-1],end],[[1:SldDist(2):end-1],end],[[1:SldDist(3):end-1],end]) = 1;
  else  % 2D
     idxMat([[1:SldDist(1):end-1],end],[[1:SldDist(2):end-1],end]) = 1;
  end

  idx = find(idxMat); 
  [rows, cols, deps] = ind2sub(size(idxMat),idx);  clear idxMat idx
  c = cell(numBlock, 1);          % index of patches in each cluster
  for k = 1 : numBlock
      c{k,1} = find(vIdx == k);  
  end

  TotalCluMap = zeros(numBlock, prod(ImgSiz), 'single');
  for k = 1 : numBlock
      ClusterMap_k = zeros(ImgSiz, 'single');
      if length(ImgSiz) == 3  
         for ii = 1 : PatNum                
             if( ~isempty(find( c{k,1} ==ii,1) ) )
                ClusterMap_k(rows(ii):rows(ii)+PatSiz(1)-1,cols(ii):cols(ii)+PatSiz(2)-1,deps(ii):deps(ii)+PatSiz(3)-1) ...
              = ClusterMap_k(rows(ii):rows(ii)+PatSiz(1)-1,cols(ii):cols(ii)+PatSiz(2)-1,deps(ii):deps(ii)+PatSiz(3)-1) + 1;
             end              
          end  
      else
         for ii = 1 : PatNum                
             if( ~isempty(find( c{k,1} ==ii,1) ) )
                ClusterMap_k(rows(ii):rows(ii)+PatSiz(1)-1,cols(ii):cols(ii)+PatSiz(2)-1) ...
              = ClusterMap_k(rows(ii):rows(ii)+PatSiz(1)-1,cols(ii):cols(ii)+PatSiz(2)-1) + 1;
             end              
         end  
      end  
      TotalCluMap(k,:) = col(ClusterMap_k);
  end 
  clear ClusterMap_k
  [~, ClusterMap] = max(TotalCluMap ,[], 1); 
  ClusterMap = single(ClusterMap);
  ClusterMap = reshape(ClusterMap, ImgSiz);


end


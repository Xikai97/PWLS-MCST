function [z, cost] = pwls_sqs_os_mom2(x, Ab, yi, wi, R, denom, D_R, niter, pixmax, chat)
%| penalized weighted least squares estimation/reconstruction
%| using separable quadaratic surrogates algorithm with
%| (optionally relaxed) ordered subsets + Momentum.  (relaxation ensures convergence.)
%|
%| Reference: Table IV in "Combining Ordered Subsets and Momentum for 
%| Accelerated X-Ray CT Image Reconstruction"
%|
%| cost(x) = (y-Gx)' W (y-Gx) / 2 + R(x)
%|
%| in
%|	x	[np 1]		initial estimate
%|	Ab	[nd np]		Gblock object (needs abs(Ab) method)
%|			         	or sparse matrix (implies nsubset=1)
%|	yi	[nb na]		measurements (noisy sinogram data)
%|	wi	[nb na]		weighting sinogram (default: [] for uniform)
%|	denom	[np 1]		precomputed denominator
%|	R			    penalty object 
%|	niter			# of iterations (including 0)
%|
%| optional
%|	pixmax	[1] or [2]	max pixel value, or [min max] (default [0 inf])
%|	aai	[ns nt na]	precomputed row sums of |Ab|
%|	relax0	[1] or [2]	relax0 or (relax0, relax_rate)
%|	chat
%|
%| out
%|	x	[np 1]		last iterate
%|	cost	[niter 1]	cost values
%|
%| Copyright 2002-2-12, Jeff Fessler, University of Michigan
%| Copyright 2015-8-13, Donghwan Kim, University of Michigan
%| 2018-11-24, implenmented by Xuehang Zheng, UM-SJTU Joint Inistitute


if nargin < 4, help(mfilename), error(mfilename), end

Ab = block_op(Ab, 'ensure'); % make it a block object (if not already)
nblock = block_op(Ab, 'n');
starts = subset_start(nblock);

cpu etic

if ~isvar('niter')	|| isempty(niter),	niter = 1;	end
if ~isvar('pixmax')	|| isempty(pixmax),	pixmax = inf;	end
if ~isvar('chat')	|| isempty(chat),	chat = false;	end
if isempty(wi)
	wi = ones(size(yi), class(yi));
end

if ~isvar('relax0') || isempty(relax0)
	relax0 = 1;
end
if length(relax0) == 1
	relax_rate = 0;
elseif length(relax0) == 2
	relax_rate = relax0(2);
	relax0 = relax0(1);
else
	error relax
end

if length(pixmax) == 2
	pixmin = pixmax(1);
	pixmax = pixmax(2);
elseif length(pixmax) == 1
	pixmin = 0;
else
	error pixmax
end

[nb na] = size(yi);
x = x(:);
x = max(x,pixmin);
x = min(x,pixmax);

% initialization
den = denom + D_R;
den(denom == 0) = 0; %
t = 1; z0 = x; z = x;
num_accu = 0;    t_accu = 0; % accumulated values



% loop over iterations
for iter = 1:niter
	ticker(mfilename, iter, niter)

	relax = relax0 / (1 + relax_rate * (iter-1));
	
    % loop over subsets
	for iset=1:nblock
		iblock = starts(iset);
		ia = iblock:nblock:na;

		li = Ab{iblock} * z;
		li = reshape(li, nb, length(ia));
		resid = wi(:,ia) .* (yi(:,ia) - li);
		grad = Ab{iblock}' * resid(:); % G' * W * (y - G*x)

		num = nblock * grad - R.cgrad(x);
	      
        num_accu = num_accu + t * num;
              
        t = 0.5*(1 + sqrt(1+4*t^2));
        t_accu = t_accu + t;
        
		x = z + relax * div0(num, den);     % relaxed update
	    x = max(x,pixmin);              % lower bound
        x = min(x,pixmax);              % upper bound

        v = z0 + div0(num_accu, den);
 	    v = max(v,pixmin);              % lower bound
        v = min(v,pixmax);              % upper bound       
        z = x + t/t_accu * (v - x);

    end

    if chat
        % calculate the cost value for each iteration
        df = .5 * sum(wi(:) .* (yi(:) - Ab * z).^2, 'double');
        reg =  R.penal(Ab, z, wi, yi);
        cost(:,iter) =  df + reg;
    else
        cost = 0;
    end
    
end




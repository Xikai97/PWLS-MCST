function [x, costt] = pwls_sqs_os_mom(x, Ab, yi, wi, R, denom, D_R, niter, pixmax, ...
		  mom, chat)
%| penalized weighted least squares estimation/reconstruction
%| using separable quadaratic surrogates algorithm with
%| (optionally relaxed) ordered subsets.  (relaxation ensures convergence.)
%| + Nesterov's fast gradient method
%| + Kim and Fessler's optimal gradient method [kim::ofo]
%|
%| cost(x) = (y-Gx)' W (y-Gx) / 2 + R(x)
%|
%| in
%|	x	[np 1]		initial estimate
%|	Ab	[nd np]		Gblock object (needs abs(Ab) method)
%|			         	or sparse matrix (implies nsubset=1)
%|	yi	[nb na]		measurements (noisy sinogram data)
%|	wi	[nb na]		weighting sinogram (default: [] for uniform)
%|	R			penalty object (see Robject.m)
%|	niter			# of iterations (including 0)
%|
%| optional
%|	pixmax	[1] or [2]	max pixel value, or [min max] (default [0 inf])
%|	denom	[np 1]		precomputed denominator
%|	aai	[ns nt na]	precomputed row sums of |Ab|
%|	relax0	[1] or [2]	relax0 or (relax0, relax_rate)
%|	conv	[np 1]		converged image
%|	mom	[1]		0 if no, 
%|				1 if Nesterov's, 
%|				2 if Kim and Fessler's,
%|	chat
%|
%| out
%|	x	[np 1]		last iterate
%|	cost	[niter 1]	cost values
%|
%| Copyright 2002-2-12, Jeff Fessler, University of Michigan
%| Copyright 2015-8-13, Donghwan Kim, University of Michigan

if nargin < 4, help(mfilename), error(mfilename), end

Ab = block_op(Ab, 'ensure'); % make it a block object (if not already)
nblock = block_op(Ab, 'n');
starts = subset_start(nblock);

cpu etic

if ~isvar('mom')	|| isempty(mom),	mom = 0;	end

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
% for mom
if (mom == 1)
	tprev = 1; xprev = x;
elseif (mom == 2)
	tprev = 1; xprev = x; xprev0 = x;
end

costt = [];
% loop over iterations
for iter = 1:niter
	ticker(mfilename, iter, niter)

	relax = relax0 / (1 + relax_rate * (iter-1));
	
    % loop over subsets
	for iset=1:nblock
		iblock = starts(iset);
		ia = iblock:nblock:na;

		li = Ab{iblock} * x;
		li = reshape(li, nb, length(ia));
		resid = wi(:,ia) .* (yi(:,ia) - li);
		grad = Ab{iblock}' * resid(:); % G' * W * (y - G*x)

		num = nblock * grad - R.cgrad(x);
		den = denom + D_R;
		den(denom == 0) = 0; %		

		x = x + relax * div0(num, den);     % relaxed update
	        x = max(x,pixmin);              % lower bound
        	x = min(x,pixmax);              % upper bound
    
%        tmp = R.penal(Ab, x, wi, yi);
%        costt(iter) = tmp(1);
            
        if( ~( iter == niter && iset == nblock) )   
		if (mom == 1) 
			t = 1/2*(1 + sqrt(1+4*tprev^2));
                        xtmp = x;
                        x = x + (tprev - 1)/t * (x - xprev);
                        xprev = xtmp;
                        tprev = t;
                elseif (mom == 2)
			t = 1/2*(1 + sqrt(1+4*tprev^2));
                        xtmp = x;
                        x = x + (tprev - 1)/t * (x - xprev) ...
				+ tprev/t * (x - xprev0);
                        xprev = xtmp; 
			xprev0 = x;
                        tprev = t;
		end
        end    

    end

end

costt = costt(:);

if chat
  % calculate the cost value for each outer iteration
   cost = R.penal(Ab, x, wi, yi);
else
   cost = 0;
end

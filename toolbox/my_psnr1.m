function psnrval = my_psnr1( I, ref, peakval )

err = norm(I(:)-ref(:), 2).^2 / numel(I);

psnrval = 10*log10(peakval.^2/err);

end
fp = fopen('output.txt','w+');
fprintf(fp, 'image mean_l var_l entropy e_r e_g e_b clip tile PSNR SSIM pre_PSNR pre_SSIM\n');

for it = 1:100
   ground_truth = sprintf('img/%d-clear.png', it);
   hazy = sprintf('img/%d.png', it);
   l_AOD = sprintf('img/%d_AOD-Net.png', it);
   
   L = imread(ground_truth);
   H = imread(hazy);
   AOD = imread(l_AOD);
   
   %[peaksnr, ~] = psnr(H, L); 
   %fprintf('\n Haze PSNR(%d) = %0.4f', it, peaksnr);   
   [pre_PSNR, ~] = psnr(AOD, L); 
   %fprintf('\n AOD PSNR(%d) = %0.4f', it, peaksnr);  
   %[ssimval, ~] = ssim(H, L); 
   %fprintf('\n Haze SSIM(%d) = %0.4f', it, ssimval); 
   [pre_SSIM, ~] = ssim(AOD, L); 
   %fprintf('\n AOD SSIM(%d) = %0.4f', it, ssimval);
   
   
   gray = rgb2gray(AOD);
   % mean luminance of grayscale image
   ml = mean2(gray);
   % something like variance?
   vl = var(double(gray(:)))/255;
   
   % entropy
   e = entropy(gray);
   e_r = entropy(AOD(:,:,1));
   e_g = entropy(AOD(:,:,2));
   e_b = entropy(AOD(:,:,3));

   
   clip = 0.1;
   for ii = 1:10
       gs = 2;
       for jj = 1:11
           l_I = sprintf('img/clahe/%d_%0.1f-%d.png', it, clip, gs);
           I = imread(l_I);
           [peaksnr, snr] = psnr(I, L); 
           fprintf('\n %d_%0.1f-%d PSNR = %0.4f', it, clip, gs, peaksnr);
           
           [ssimval, ~] = ssim(I, L); 
           fprintf('\n %d_%0.1f-%d SSIM = %0.4f', it, clip, gs, ssimval);

           fprintf(fp, '%d %0.4f %0.4f %0.4f %0.4f %0.4f %0.4f %0.1f %d %0.4f %0.4f %0.4f %0.4f\n', ...
                        it, ml, vl, e, e_r, e_g, e_b, clip, gs, peaksnr, ssimval, pre_PSNR, pre_SSIM);
           
           gs = gs + 2;
       end
       clip = clip + 0.1;
   end
end
fclose(fp);
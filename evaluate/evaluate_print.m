p_H = 0.0;
p_AOD = 0.0;
s_H = 0.0;
s_AOD = 0.0;

fp = fopen('clahe_results.txt','w+');
fprintf(fp, 'image# mean_luminance_Haze Mean_luminance_AOD clip gridsize PSNR SSIM\n');

for it = 1:18
   ground_truth = sprintf('img/%d-clear.png', it);
   hazy = sprintf('img/%d.png', it);
   l_AOD = sprintf('img/%d_AOD-Net.png', it);
   
   L = imread(ground_truth);
   H = imread(hazy);
   AOD = imread(l_AOD);
   
   [peaksnr, ~] = psnr(H, L); 
   fprintf('\n Haze PSNR(%d) = %0.4f', it, peaksnr);
   p_H = p_H + peaksnr;
   
   [peaksnr, ~] = psnr(AOD, L); 
   fprintf('\n AOD PSNR(%d) = %0.4f', it, peaksnr);
   p_AOD = p_AOD + peaksnr;
   
   [ssimval, ~] = ssim(H, L); 
   fprintf('\n Haze SSIM(%d) = %0.4f', it, ssimval);
   s_H = s_H + ssimval;
   
   [ssimval, ~] = ssim(AOD, L); 
   fprintf('\n AOD SSIM(%d) = %0.4f', it, ssimval);
   s_AOD = s_AOD + ssimval;
   
   ml_H = mean2(rgb2gray(H));
   ml_AOD = mean2(rgb2gray(AOD));

   
   clip = 0.4;
   for ii = 1:5
       gs = 4;
       for jj = 1:4
           l_I = sprintf('img/clahe/%d_%0.1f-%d.png', it, clip, gs);
           I = imread(l_I);
           [peaksnr, snr] = psnr(I, L); 
           fprintf('\n %d_%0.1f-%d PSNR = %0.4f', it, clip, gs, peaksnr);
           
           [ssimval, ~] = ssim(I, L); 
           fprintf('\n %d_%0.1f-%d SSIM = %0.4f', it, clip, gs, ssimval);
           
           %mean_lum_H = mean2(rgb2gray(H))
           %mean_lum_AOD = mean2(rgb2gray(AOD))

           fprintf(fp, '%d %0.4f %0.4f %0.1f %d %0.4f %0.4f\n', it, ml_H, ml_AOD, clip, gs, peaksnr, ssimval);
           
           gs = gs + 4;
       end
       clip = clip + 0.1;
   end
end
fclose(fp);
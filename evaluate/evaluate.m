for ii = 1:9
   ground_truth = sprintf('img/%d_clear.jpg', ii);
   hazy = sprintf('img/%d.jpg', ii);
   AOD = sprintf('img/%d_AOD-Net.jpg', ii);
   DARK = sprintf('img/%d_DarkChannel.jpg', ii);
   AD = sprintf('img/%d_AOD->DARK.jpg', ii);

   L = imread(ground_truth);
   H = imread(hazy);
   Im_AOD = imread(AOD);
   Im_DARK = imread(DARK);
   Im_AD = imread(AD);
   
   [peaksnr, ~] = psnr(H, L); 
   fprintf('\n Haze PSNR(%d) = %0.4f', ii, peaksnr);
   [peaksnr, ~] = psnr(Im_AOD, L); 
   fprintf('\n AOD PSNR(%d) = %0.4f', ii, peaksnr);
   [peaksnr, ~] = psnr(Im_DARK, L); 
   fprintf('\n DARK PSNR(%d) = %0.4f', ii, peaksnr);
   [peaksnr, ~] = psnr(Im_AD, L); 
   fprintf('\n AOD->DARK PSNR(%d) = %0.4f', ii, peaksnr);
   
   [ssimval, ~] = ssim(H, L); 
   fprintf('\n Haze SSIM(%d) = %0.4f', ii, ssimval);
   [ssimval, ~] = ssim(Im_AOD, L); 
   fprintf('\n AOD SSIM(%d) = %0.4f', ii, ssimval);
   [ssimval, ~] = ssim(Im_DARK, L); 
   fprintf('\n DARK SSIM(%d) = %0.4f', ii, ssimval);
   [ssimval, ~] = ssim(Im_AD, L); 
   fprintf('\n AOD->DARK SSIM(%d) = %0.4f\n', ii, ssimval);
end
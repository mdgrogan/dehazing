p_H = 0.0;
p_AOD = 0.0;
p_I1 = 0.0;
p_I2 = 0.0;
p_I3 = 0.0;
p_I4 = 0.0;
s_H = 0.0;
s_AOD = 0.0;
s_I1 = 0.0;
s_I2 = 0.0;
s_I3 = 0.0;
s_I4 = 0.0;

for ii = 1:18
   ground_truth = sprintf('img/%d-clear.png', ii);
   hazy = sprintf('img/%d.png', ii);
   l_AOD = sprintf('img/%d_AOD-Net.png', ii);
   l_I1 = sprintf('img/clahe/%d_CLAHE-0.4-8.png', ii);
   l_I2 = sprintf('img/clahe/%d_CLAHE-0.4-12.png', ii);
   l_I3 = sprintf('img/clahe/%d_CLAHE-0.6-8.png', ii);
   l_I4 = sprintf('img/clahe/%d_CLAHE-0.6-12.png', ii);


   L = imread(ground_truth);
   H = imread(hazy);
   AOD = imread(l_AOD);
   I1 = imread(l_I1);
   I2 = imread(l_I2);
   I3 = imread(l_I3);
   I4 = imread(l_I4);

   [peaksnr, ~] = psnr(H, L); 
   fprintf('\n Haze PSNR(%d) = %0.4f', ii, peaksnr);
   p_H = p_H + peaksnr;
   [peaksnr, ~] = psnr(AOD, L); 
   fprintf('\n AOD PSNR(%d) = %0.4f', ii, peaksnr);
   p_AOD = p_AOD + peaksnr;
   [peaksnr, ~] = psnr(I1, L); 
   fprintf('\n CLAHE-0.4-8 PSNR(%d) = %0.4f', ii, peaksnr);
   p_I1 = p_I1 + peaksnr;
   [peaksnr, ~] = psnr(I2, L); 
   fprintf('\n CLAHE-0.4-12 PSNR(%d) = %0.4f', ii, peaksnr);
   p_I2 = p_I2 + peaksnr;
   [peaksnr, ~] = psnr(I3, L); 
   fprintf('\n CLAHE-0.6-8 PSNR(%d) = %0.4f', ii, peaksnr);
   p_I3 = p_I3 + peaksnr;
   [peaksnr, ~] = psnr(I4, L); 
   fprintf('\n CLAHE-0.6-12 PSNR(%d) = %0.4f', ii, peaksnr);
   p_I4 = p_I4 + peaksnr;
   
   [ssimval, ~] = ssim(H, L); 
   fprintf('\n Haze SSIM(%d) = %0.4f', ii, ssimval);
   s_H = s_H + ssimval;
   [ssimval, ~] = ssim(AOD, L); 
   fprintf('\n AOD SSIM(%d) = %0.4f', ii, ssimval);
   s_AOD = s_AOD + ssimval;
   [ssimval, ~] = ssim(I1, L); 
   fprintf('\n CLAHE-0.4-8 SSIM(%d) = %0.4f', ii, ssimval);
   s_I1 = s_I1 + ssimval;
   [ssimval, ~] = ssim(I2, L); 
   fprintf('\n CLAHE-0.6-12 SSIM(%d) = %0.4f', ii, ssimval);
   s_I2 = s_I2 + ssimval;
   [ssimval, ~] = ssim(I3, L); 
   fprintf('\n CLAHE-0.4-8 SSIM(%d) = %0.4f', ii, ssimval);
   s_I3 = s_I3 + ssimval;
   [ssimval, ~] = ssim(I4, L); 
   fprintf('\n CLAHE-0.6-12 SSIM(%d) = %0.4f\n', ii, ssimval);
   s_I4 = s_I4 + ssimval;
end

fprintf('\n Haze PSNR = %0.4f', p_H/18);
fprintf('\n AOD PSNR = %0.4f', p_AOD/18);
fprintf('\n CLAHE-0.4-8 PSNR = %0.4f', p_I1/18);
fprintf('\n CLAHE-0.4-12 PSNR = %0.4f', p_I2/18);
fprintf('\n CLAHE-0.6-8 PSNR = %0.4f', p_I3/18);
fprintf('\n CLAHE-0.6-12 PSNR = %0.4f', p_I4/18);
fprintf('\n Haze SSIM = %0.4f', s_H/18);
fprintf('\n AOD SSIM = %0.4f', s_AOD/18);
fprintf('\n CLAHE-0.4-8 SSIM = %0.4f', s_I1/18);
fprintf('\n CLAHE-0.6-12 SSIM = %0.4f', s_I2/18);
fprintf('\n CLAHE-0.4-8 SSIM = %0.4f', s_I3/18);
fprintf('\n CLAHE-0.6-12 SSIM = %0.4f\n', s_I4/18);










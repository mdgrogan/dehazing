%% Single Image Haze Removal Using Dark Channel Prior - by Shiyu Dong and Yilin Yang
%clear; close all; clc;

for i = 1 : 9
    tic;
    %% read image
    try        
        img_file = ['../img/', num2str(i), '_AOD-Net.jpg'];        
        img = imread(img_file);
    catch
        img_file = ['../img/', num2str(i), '.png'];        
        img = imread(img_file);
    end
    %img = imresize(img, [500, NaN]);

    %% dehaze
    [J, t_, t] = dehaze(img);
    imwrite(J, ['../img/', num2str(i), '_AOD->DARK.jpg']);
    %imwrite(t_, ['../results/', num2str(i), '_DarkChannel_TransRaw.png']);
    %imwrite(t, ['../results/', num2str(i), '_DarkChannel_Trans.png']);
    time = toc;
    disp(['Image ', num2str(i), ' saved. Time: ', num2str(time), 's. ']);
end

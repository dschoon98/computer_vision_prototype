
addpath .



myFolder = '/home/morgoth/computer_vision_prototype/bebop_images/cyberzoo_poles_panels/20190121-140205' %20190121-140205
writeFolder = '/home/morgoth/computer_vision_prototype/CV_output/resize_no_Gauss/'
%I = imread('/home/morgoth/computer_vision_prototype/bebop_images/cyberzoo_poles/20190121-135009/85878042.jpg'); % i dont understand paths
filePattern = fullfile(myFolder, '*.jpg');
theFiles = dir(filePattern);


for k = 1:length(theFiles)

jpgFileName = strcat(myFolder, num2str(k), '.jpg');
baseFileName = theFiles(k).name;
fullFileName = fullfile(theFiles(k).folder, baseFileName);	  
I = imread(fullFileName);
bw = rgb2gray(I);
bw = bw' ;
Im_smol = imresize(bw, 0.1) ; % check if resizing is bad, its not downsampled 


G_Im10 = imgaussfilt(Im_smol, 2);
[edge_im , im_min, im_max , im_med] = maximum_pix(bw, 10);



Gauss_diff = Im_smol - G_Im10;
idx = find(Gauss_diff > 10);
Gauss_diff(idx) = 255 ;


[morph_edge_min, morph_edge_max] = grab_lines_2(edge_im , 15 );
together_im = [bw ; edge_im; morph_edge_max] ;
new_name = strcat(writeFolder , num2str(k) , '.jpg');
imwrite(together_im, new_name);
 
end

 
% the best ones here are Gauss_lines_min or Gauss_lines_max. We should pick
% one.

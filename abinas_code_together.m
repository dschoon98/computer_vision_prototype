addpath .

tic
I = imread('/home/morgoth/computer_vision_prototype/images/flag_and_multi_color.jpeg'); % i dont understand paths
bw = rgb2gray(I);
% may want subsample step ?
% here's the subsampling you guys do:
tic
Im_smol = imresize(bw, 0.1)

figure
imshow(bw)

%method #1 : Gaussian filter

G_Im10 = imgaussfilt(bw, 2);
Gauss_diff = bw - G_Im10;
idx = find(Gauss_diff > 10);
Gauss_diff(idx) = 255 ;
[Gauss_lines_min, Gauss_lines_max ] = grab_lines_2(Gauss_diff , 15 );
toc

% the best ones here are Gauss_lines_min or Gauss_lines_max. We should pick
% one.

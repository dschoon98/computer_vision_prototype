
addpath . 

% grab the image, convert to black and white , transpose since everything
% was sideways
filename = '/home/morgoth/computer_vision_prototype/bebop_images/cyberzoo_poles_panels/20190121-140205/71849392.jpg'
I = imread(filename);
bw = rgb2gray(I);
bw = bw' ;


% filter with a small gaussian ( possible with opencv , may need to adjust gauss) 
G_Im10 = imgaussfilt(bw, 2);

% subtract the gaussian from the original image
Gauss_diff = bw - G_Im10;

% set the values above a certain threshold to maximum so I can see things
% better
idx = find(Gauss_diff > 10);
Gauss_diff(idx) = 255 ;

% a second edge detector ( see if its neccessary) to amplify vertical lines
[morph_edge_min, morph_edge_max] = grab_lines_2(edge_im , 15 );
figure
imshow(morph_edge_min)
% the output you want is "min" 
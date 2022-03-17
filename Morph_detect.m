%% test code : 
addpath .
I = imread('/home/morgoth/computer_vision_prototype/bebop_images/cyberzoo_poles_panels/20190121-140205/71849392.jpg') ;
%I = imread('/home/morgoth/computer_vision_prototype/bebop_images/cyberzoo_poles_panels/20190121-140205/71849392.jpeg'); % i dont understand paths
Im_black = rgb2gray(I);
Im_black = Im_black' ;
% as expected we amplify noise with this alg. so first I gaussian filter (
% but do not downsample)


G_Im10 = imgaussfilt(Im_black, 10);
[edge_im , im_min, im_max , im_med] = maximum_pix(Im_black, 10);
%[edge_im_gauss , im_min_gauss, im_max_gauss , im_med_gauss] = maximum_pix(edge_im, 4);
[im_min_2nd, im_max_2nd] = grab_lines_2(edge_im, 10);

figure
imshow(edge_im) ; 





% I was able to reduce noise here by taking a gaussian here :
% since we're frustrated, we can also caluclate a derivative using
% Gaussians : 

%Gauss_edge = Im_black - G_Im10 ; % this is nice 

%imshow(G_derv)

%expensive = double(Gauss_edge) .* edge_im ;
%[edge_im_exp , im_min_exp, im_max_exp , im_med_exp] = maximum_pix(expensive, 7);
%best_edges = im_max_exp ; 

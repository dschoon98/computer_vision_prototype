function [edge_im, im_min, im_max, im_med] = maximum_pix(Im, kern)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here

sz = size(Im)
im_min = zeros(sz(1), sz(2));
im_max = zeros(sz(1),sz(2));


for i = 1:sz(1) - kern 
    for j = 1:sz(2) - kern 

        kernel = Im(i:i+kern, j:j+kern) ; 
        
        im_max(i,j) = max(kernel, [], 'all');
        im_min(i,j) = min(kernel, [], 'all');
        im_med(i,j) = median(kernel,'all');

    end
end

Im = double(Im);
edge_im = Im - 0.5*(im_max + im_min) ; % may need to normalize 

end
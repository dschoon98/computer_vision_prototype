function [ im_min, im_max ] = grab_lines_2(Im, length) 

sz = size(Im);
im_min = zeros(sz(1), sz(2));
im_max = zeros(sz(1),sz(2));

for i = 1:sz(1) - length 
    for j = 1:sz(2)

    kernel = Im(i:i+length , j) ; 
    im_max(i, j) = max(kernel, [], "all");
    im_min(i, j) = min(kernel, [], "all");

    end 
end
end 

% note theres a dark band on the floor. remove it when you have a better
% sense of what happens next

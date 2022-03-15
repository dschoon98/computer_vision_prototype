

% Abinas OF algorithm
function [line_vect] = grab_lines(Im, thresh_im , line_length , slant )

% threshold the image by value and find those indexes 
thresh_im = find(Im > thresh_im) ;
[row, col] = ind2sub(size(Im), thresh_im);

IO = [row, col] ; 
line_vect = {} ;

% maybe jump by slant 
% select "like columns" 

for i = min(col) : max(col) - slant 

    % only select the columns in a certain range  
    col_idx = find(i <= col & col <= i + slant) ;
    size(col_idx)
    vect = IO( col_idx, : )  
    size(vect)
    sz = size(col_idx); 

    if sz(1) > line_length  % thresholds the length of the line
        line_vect{i} = vect ;
    end

end
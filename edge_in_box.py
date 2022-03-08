import numpy as np
import imageio as iio
import matplotlib
import pandas as pd
matplotlib.use('Agg')
import matplotlib.pyplot as plt


%
%
# I expect a "box", which is just a matrix and a "kernel" which will define edge size


# erode takes minimum valu
def black_and_white(img):

    dim = np.shape(img) ;
    x = np.arange(0, dim[0], 1) ;
    y = np.arange(0 , dim[1], 1)
# sum all luminance
    black = [[np.sum(img[v][i][:3]) for i in y] for v in x];
# normalize back into a 255 range
    maxi = np.max(black)
    mult = 255/maxi
    black = [[black[v][i] * mult  for i in y] for v in x] ;

    return black


def edge_box(box, kern , black_white):


# turns to grayscale image using different techniques (true grayscale, pick one color) :
    if black_white == 1 :
        black_box = black_and_white(box) ;  # may be more efficient just to chose 1 color gradient
    elif black_white == 2 :
        black_box = [[img[v][i][0] for i in y] for v in x] ;
    elif black_white == 3 :
        black_box = [[img[v][i][1] for i in y] for v in x] ;
    elif black_white == 4 :
        black_box = [[img[v][i][2] for i in y] for v in x];


    dim = np.shape(box)
    x = np.arange(0, dim[0] - kern, 1); # I'm going to be adding the kern so...
    y = np.arange(0, dim[1] - kern, 1); # I'm going to be adding the kern so...
    black_df = pd.DataFrame(black_box)

    di = [[np.max(black_df[x[v:v+kern]][y[i:i+kern]]) for i in y] for v in x]; #dilates ( some problem, convert to dataframe)
    ero = [[np.min(black_df[x[v:v+ kern]][y[i: i + kern]]) for i in y] for v in x]; # erodes (same deal)

    lines = di - ero ;
    return lines

img = iio.imread("/home/morgoth/Desktop/kandinsky_portrait.png")

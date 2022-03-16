import numpy as np
import matplotlib.pyplot as plt
import cv2
resize_factor = 10


def filter_color(im, y_low, y_high, u_low, u_high, v_low, v_high, resize_factor):
    YUV = cv2.cvtColor(im, cv2.COLOR_BGR2YUV);
    Filtered = np.zeros([YUV.shape[0], YUV.shape[1]]);
    for y in range(YUV.shape[0]):
        for x in range(YUV.shape[1]):
            if(YUV[y,x,0] >= y_low and YUV[y,x,0] <= y_high and \
               YUV[y,x,1] >= u_low and YUV[y,x,1] <= u_high and \
               YUV[y,x,2] >= v_low and YUV[y,x,2] <= v_high):
                Filtered[y,x] = 1;
    return Filtered


def edge_definer(bin_mat):
    rows = len(bin_mat[:,1])
    cols = len(bin_mat[1,:])
    matrix_edge = np.zeros([rows,cols])
    for i in range(rows):
        temp_val = 0
        for j in range(cols):
            if temp_val != bin_mat[i,j]:
                matrix_edge[i,j] = 1
                if temp_val == 0:
                    temp_val = 1
                else:
                    temp_val = 0
    for j in range(cols):
        temp_val = 0
        for i in range(rows):
            if temp_val != bin_mat[i,j]:
                matrix_edge[i,j] = 1
                if temp_val == 0:
                    temp_val = 1
                else:
                    temp_val = 0
    return matrix_edge
def edge_finder(im):
    bin_mat1 = filter_color(im,180,253,100,150,130,140,resize_factor)  #orange pole and chairs
    bin_mat2 = filter_color(im,70,120,150,160,100,120,resize_factor)   #Blue chair
    bin_mat3 = filter_color(im,100,200,90,130,160,240,resize_factor) #Orange 
    
    bin_mat_tot = bin_mat1+bin_mat2+bin_mat3   
    matrix_edge = edge_definer(bin_mat_tot)
    
    plt.figure()
    plt.imshow(matrix_edge)
    plt.title('Edges only')     
    return matrix_edge

#image_name = 'images/image1.jpeg'
#im = cv2.imread(image_name);
#im = cv2.resize(im, (int(im.shape[1]/resize_factor), int(im.shape[0]/resize_factor)));

#im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB);
#plt.figure()
#plt.imshow(im_rgb)
#plt.title('Resized image')
#
#edge_finder(im)
#
#
#
#im = cv2.GaussianBlur(im,(3,3),0)
#im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB);
#plt.figure()
#plt.imshow(im_rgb)
#plt.title('With blur')
#
#edge_finder(im)
#
#


            
    
    





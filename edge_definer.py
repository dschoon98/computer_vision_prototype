import numpy as np
import matplotlib.pyplot as plt
import cv2

resize_factor = 10

def filter_color(im, y_low, y_high, u_low, u_high, v_low, v_high):
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
def edge_finder(im,orange,blue,white,green):
    omega = np.random.uniform(0,1)
    print('omega = ',omega)
    # Orange poles   
    if omega <= orange:
        bin_mat = filter_color(im,50,170,90,130,160,240) # Orange poles, tested on bebop images
    #Blue chair
    if omega > orange and omega<= orange + blue:
        bin_mat = filter_color(im,70,120,150,160,100,120)
    if omega > orange + blue and omega <= orange + blue + white:
#        bin_mat = filter_color(im,180,253,100,150,130,160)  #White Flags NORMAL PIC, rails, qr code whites, parts of multicolor 
        bin_mat = filter_color(im,80,253,130,180,130,180)   # White flag based on bebop images
    if omega > orange + blue + white and omega <= orange + blue + white + green:
        bin_mat = filter_color(im,80,180,50,100,100,150)  # Green plants
    
    matrix_edge = edge_definer(bin_mat)
      
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


            
    
    





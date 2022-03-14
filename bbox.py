import numpy as np
import edge_definer as edge
import cv2
import matplotlib.pyplot as plt



Matrix_test3 = np.matrix([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
])

###################### 
#Image processing

resize_factor = 20
image_name = 'images/image1.jpeg'
im = cv2.imread(image_name);
im = cv2.resize(im, (int(im.shape[1]/resize_factor), int(im.shape[0]/resize_factor)));
im = cv2.GaussianBlur(im,(5,5),0)
im_rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB);
plt.figure()
plt.imshow(im_rgb)
plt.title('Resized image with blur')

matrix_edge = edge.edge_finder(im)

####################

Matrix_test3 = matrix_edge  #Now the matrix is copied like 4 times, this should change! 

object_amount =100

object_matrix = np.zeros([object_amount, 8])
bin_mat = Matrix_test3
switch_var = 1
rows = bin_mat.shape[0]
cols = bin_mat.shape[1]

k=0
going_left=1
hit=0

def scanning(i_start, i_end, j_start,j_end, bin_mat):
    global k
    global object_matrix
    global hit
    stopvar=0
    
    for i in range(int(i_start), int(i_end)):
        for j in range(int(j_start), int(j_end)):
            if bin_mat[i,j] == 1:
                object_matrix[k, 0] =i
                object_matrix[k, 1] =j
                stopvar=1
                hit=1
                break
        if not stopvar:
            hit=0
        if stopvar:
            right_maxima_finder(bin_mat, object_matrix[k,0], object_matrix[k,1])
            left_maxima_finder(bin_mat, object_matrix[k,0], object_matrix[k,1])
            lower_maxima_finder(bin_mat, object_matrix[k,2], object_matrix[k,3])
            break
            
        
def x_ray(bin_mat):
    global object_matrix
    global hit
    global object_amount
    i_start=0
    i_end=rows
    j_start=0
    j_end=cols
    global k
    stopvar=0
    running=1

    while running:
        stopvar=0
        for i in range(int(i_start), int(i_end)):
            xr=iter(range(int(j_start), int(j_end)))
            for j in xr:
                for K in range(k):
                    if i>=object_matrix[K,0]-1 and i<=object_matrix[K,6]+1 and j>=object_matrix[K,5]-1 and j<=object_matrix[K,3]+1:
                        for z in range(int((object_matrix[K,3]-object_matrix[K,5])+1)):
                            next(xr)
            
                
                if bin_mat[int(i),int(j)]==1:
                    object_matrix[k,0]=i
                    object_matrix[k,1]=j
                    stopvar=1
                    break
            if stopvar:
                right_maxima_finder(bin_mat, object_matrix[k,0], object_matrix[k,1])
                left_maxima_finder(bin_mat, object_matrix[k,0], object_matrix[k,1])
                lower_maxima_finder(bin_mat, object_matrix[k,2], object_matrix[k,3])
                i_start=object_matrix[k,0]
                k+=1
                break
            if j==cols-1 and i==rows-1:
                running=0
 
#This function finds the right most boundary of an object starting from the coordinates of the most upper point of the object
def right_maxima_finder(bin_mat, i, j):
    global k
    global object_matrix;
    
    while True:
        if j==cols-1:
            break
        if bin_mat[int(i), int(j+1)] == 1:
            j+=1
        elif bin_mat[int(i+1), int(j+1)] == 1:
            i+=1
            j+=1
        elif bin_mat[int(i+1), int(j)] == 1:
            i+=1
        else:
            break
    object_matrix[k,2]=i
    object_matrix[k,3]=j



    

#This function finds the left most edge of the object from the most upper coordinate of the object edge.
def left_maxima_finder(Matrix_edges, i, j, edge_gap = 0):
    global k
    global object_matrix;

    while True:
        if j==0:
            break
        if Matrix_edges[int(i), int(j-1)] == 1:
            j-=1
        elif Matrix_edges[int(i-1), int(j-1)] == 1:
            i-=1
            j-=1
        elif Matrix_edges[int(i-1), int(j)] == 1:
            i-=1
        else:
            break
    object_matrix[k,4]=i
    object_matrix[k,5]=j

    return j

def lower_maxima_finder(Matrix_edges, i_right, j_right):
    global k
    global object_matrix;

    while True:
        if i_right==rows-1:
            break
        if Matrix_edges[int(i_right+1), int(j_right)] == 1:
            i_right+=1
        elif Matrix_edges[int(i_right+1), int(j_right-1)] == 1:
            i_right+=1
            j_right-=1
        elif Matrix_edges[int(i_right),int(j_right-1)] == 1:
            j_right-=1
        else:
            break
    object_matrix[k, 6]=i_right
    object_matrix[k, 7]=j_right
    
    
    
#----Testing------


x_ray(bin_mat)

        

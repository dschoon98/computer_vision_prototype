import numpy as np
import images
import YUV_slices as YUV
import cv2

YUV.filter_color('images/image1.jpeg',50,300,0,120,160,220,resize_factor=5)

cols = 400
rows = 400
#matrix_yuv = np.zeros([rows,cols],dtype=object)
#for i in range(rows):
#    for j in range(cols):
#        temp_array=np.zeros(3)
#        for k in range(3):
#            temp_array[k] = np.random.rand()
#            
#        matrix_yuv[i,j] = temp_array
#
#print(matrix_yuv)        


#matrix_yuv = np.array([[np.array([3,2,3]),np.array([1,2,3])],[np.array([1,2,3]),np.array([1,2,3])]])

simple_matrix_y = np.zeros([rows,cols])
simple_matrix_u = np.zeros([rows,cols])
simple_matrix_v = np.zeros([rows,cols])
threshold_y = 0.5
threshold_u = 0.5
threshold_v = 0.5

def update_simple_y(matrix_yuv,threshold):
    global simple_matrix_y
    for i in range(rows):
        for j in range(cols):
            if (matrix_yuv[i,j])[0] >= threshold:
                simple_matrix_y[i,j] = 1

def update_simple_u(matrix_yuv,threshold):
    global simple_matrix_u
    for i in range(rows):
        for j in range(cols):
            if (matrix_yuv[i,j])[1] >= threshold:
                simple_matrix_u[i,j] = 1
                
def update_simple_v(matrix_yuv,threshold):
    global simple_matrix_v
    for i in range(rows):
        for j in range(cols):
            if (matrix_yuv[i,j])[2] >= threshold:
                simple_matrix_v[i,j] = 1

update_simple_y(matrix_yuv,threshold_y)
update_simple_u(matrix_yuv,threshold_u)
update_simple_v(matrix_yuv,threshold_v)



def edge_definer(simple_matrix):
    matrix_edge = np.zeros([rows,cols])
    for i in range(rows):
        temp_val = 0
        for j in range(cols):
            if temp_val != simple_matrix[i,j]:
                matrix_edge[i,j] = 1
                if temp_val == 0:
                    temp_val = 1
                else:
                    temp_val = 0
    return matrix_edge
                
matrix_edge = edge_definer(simple_matrix_y)

                
#    #if 0 is links and 1 is rechts, edge :
#    for i in range(400):
#        for j in range(400):
#            
    
    
#def update_simple_u(matrix_yuv,threshold_red,simple_matrix_y):
#    global simple_matrix_y    
#    for i in range(2):
#        for j in range(2):
#            if (matrix_yuv[i,j])[0] >= threshold_red:
#                simple_matrix_y[i,j] = 1
#                print(simple_matrix_y)
#                return(simple_matrix_y)                
#
#def update_simple_v(matrix_yuv,threshold_red,simple_matrix_y):
#    global simple_matrix_y
#    for i in range(2):
#        for j in range(2):
#            if (matrix[i,j])[0] >= threshold_red:
#                simple_matrix_y[i,j] = 1
#                print(simple_matrix_y)
#                return(simple_matrix_y)                
#




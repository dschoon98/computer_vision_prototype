import numpy as np



matrix_yuv = np.array([[np.array([3,2,3]),np.array([1,2,3])],[np.array([1,2,3]),np.array([1,2,3])]])

simple_matrix_y = np.zeros([2,2])
simple_matrix_u = np.zeros([2,2])
simple_matrix_v = np.zeros([2,2])
threshold_y = 2
threshold_u = 2
threshold_v = 2

def update_simple_y(matrix_yuv,threshold):
    global simple_matrix_y
    for i in range(2):
        for j in range(2):
            if (matrix_yuv[i,j])[0] >= threshold:
                simple_matrix_y[i,j] = 1

def update_simple_u(matrix_yuv,threshold):
    global simple_matrix_y
    for i in range(2):
        for j in range(2):
            if (matrix_yuv[i,j])[1] >= threshold:
                simple_matrix_u[i,j] = 1
                
def update_simple_v(matrix_yuv,threshold):
    global simple_matrix_y
    for i in range(2):
        for j in range(2):
            if (matrix_yuv[i,j])[2] >= threshold:
                simple_matrix_v[i,j] = 1

update_simple_y(matrix_yuv,threshold_y)
update_simple_u(matrix_yuv,threshold_u)
update_simple_v(matrix_yuv,threshold_v)

def edge_definer(simple_matrix):
    #if 0 is links and 1 is rechts, edge :
    for i in range(1)
    
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




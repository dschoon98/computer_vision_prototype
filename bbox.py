import numpy as np
import cv2
import time 
import os
import matplotlib.pyplot as plt

def load_images_from_folder(folder,binary,scale_percent = 60):
    image_sequence = []

    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            if binary:            
                im_mat = img[:,:,0]            
                # Should be deleted later on, just for testting. In Matlab code, it should be already made binary
                bin_mat = np.divide(im_mat,255*np.ones([im_mat.shape[0],im_mat.shape[1]]))
                bin_mat = bin_mat.astype(int)
    #            img = cv2.rotate(img,cv2.ROTATE_90_COUNTERCLOCKWISE)
    
                image_sequence.append(bin_mat)
            else:
                image_sequence.append(img)
    return image_sequence

####################

switch_var = 1

def x_ray(bin_mat):
    global hit
    global object_amount
    global k
    k=0
    object_amount = 400
    object_matrix = np.zeros([object_amount,8])

    rows = bin_mat.shape[0]
    cols = bin_mat.shape[1]

    i_start=0
    i_end=rows
    j_start=0
    j_end=cols
    stopvar=0
    running=1
    
    while running:
        stopvar=0
        for i in range(int(i_start), int(i_end)):
            xr=iter(range(int(j_start), int(j_end)))
            for j in xr:
                if bin_mat[int(i),int(j)]==1:
                    inside_object=0
                    for K in range(k):
                        if i>=object_matrix[K,0] and i<=object_matrix[K,6] and j>=object_matrix[K,5] and j<=object_matrix[K,3]:
                            inside_object=1
                            for z in range(int(object_matrix[K, 3]-j)):
                                next(xr)
                        if inside_object:
                            break
                    if not inside_object:
                        object_matrix[k,0]=i
                        object_matrix[k,1]=j
                        stopvar=1
                        break
            if stopvar:
                right_maxima_finder(bin_mat, object_matrix[k,0], object_matrix[k,1],object_matrix)
                left_maxima_finder(bin_mat, object_matrix[k,0], object_matrix[k,1],object_matrix)
                lower_maxima_finder(bin_mat, object_matrix[k,2], object_matrix[k,3],object_matrix)
                i_start=object_matrix[k,0]
                k+=1
                break
            if j==cols-1 and i==rows-1:
                running=0
    object_matrix= np.delete(object_matrix,np.where(~object_matrix.any(axis=1))[0], axis=0)
    return object_matrix,bin_mat
 
#This function finds the right most boundary of an object starting from the coordinates of the most upper point of the object
def right_maxima_finder(bin_mat, i, j,object_matrix):
    global k
    cols = bin_mat.shape[1]
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
def left_maxima_finder(Matrix_edges, i, j, object_matrix):
    global k

    while True:
        if j==0:
            break
        if Matrix_edges[int(i), int(j-1)] == 1:
            j-=1
        elif Matrix_edges[int(i+1), int(j-1)] == 1:
            i+=1
            j-=1
        elif Matrix_edges[int(i+1), int(j)] == 1:
            i+=1
        else:
            break
    object_matrix[k,4]=i
    object_matrix[k,5]=j

def lower_maxima_finder(Matrix_edges, i_right, j_right,object_matrix):
    global k
    rows = Matrix_edges.shape[0]
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
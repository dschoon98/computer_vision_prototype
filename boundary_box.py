import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils
#The scaling factor determines how large the boundary boxes will be around the object.
Scaling = 1;

#Rn this is filled in by hand, but it would be much nicer if the algoritm detects what the dimensions of the input matrices are.
rows=20
cols=20

#This is the amount of objects that the programm can differentiatie, if the object count is larger the matrix will also be larger.
#Its important to have a fixed amount and not dynamically allocated because we don't want to flood RAM 
object_amount = 10

#Test samples:
#This is a test matrix, where the 1's depicts the edges of the different objects
#I am planning to make more of these matrices with different object orientations
#below are only two objects present.
Matrix_test1 = np.matrix([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,0],
    [0,0,1,1,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0],
    [0,1,0,1,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0],
    [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0],
    [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
])

Matrix_test2 = np.matrix([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,1,0,0,1,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
])
#Change the description of this matrix!!!!!!!!  
object_matrix = np.zeros([object_amount, 5])



bin_mat = Matrix_test1
rows = len(bin_mat[:,1])
cols = len(bin_mat[1,:])
k = 0
i_lower = 0


#This function will loop the image until it finds an object
def loop_until_hit(rows,cols,k,i_lower):
    global object_matrix
    global bin_mat
    if k>0:
#        i_lower = object_matrix[k-1, 6]  # 6th index for i_lower
        sliced_mat = bin_mat[i_lower:,:]
    else:
        sliced_mat = bin_mat
    
    for i in range(rows):    
        for j in range(cols):
            if sliced_mat[i,j] == 1:
                break
        break
    k += 1
    return i,j,k



loop_until_hit(rows, cols, k)

[i_lower, j_lower, i_left, j_left, i_right, j_right] = maxima_finder(i,j,k)

rows = rows - i_lower


#
##SHOULD BE IMPROVED FOR FINDING OBJECTS STACKED ABOVE EACH OTHER AND WITH SAME UPPER HEIGHT (should pass test2 and a test 3 that still has to be made):
#def (Matrix_test, Scaling):
#    global boundary_matrix;
#    global object_matrix;
#
#    #This loop checks for every object what it's upper_height is
#    #And also is responisble for the the identification of the objects, the highest object will be object 1 and the one below that 2 etc. etc.
#    #Should break if the object_amount limit is reached
#    #Should also do something in the case two objects are vertically stacked
#    for i in range(rows):
#        edges_row=0
#        #Checks for every row if there are edges present, the edges will always come in pairs
#        for j in range(cols):
#            if Matrix_test[i,j]==1:
#                edges_row+=1
#        #This range below checks how many objects are found in the scene, it does this by using the given that every object has to edges on every row where it lives
#        for k in range(int(edges_row/2)):
#            #zero in this case means there is no upper_height assigned or simply not there
#            #k is in this case revering the objects, thus will write a value to the upper_limmit how k=0 (thus the first object)
#            if object_matrix[k,1]==0:
#                object_matrix[k, 0]=i
#                j = np.nonzero(Matrix_test[i,:] == 1)[1][0]
#                object_matrix[k,1]=j
#    object_matrix = object_matrix[~np.all(object_matrix == 0, axis=1)]  # Delete all rows that are only zero 
#    return object_matrix
#






                


        



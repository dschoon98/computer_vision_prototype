import numpy as np
import matplotlib.pyplot as plt

#The scaling factor determines how large the boundary boxes will be around the object.
Scaling = 1;

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
    [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,1,0,0],
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
object_matrix = np.zeros([object_amount, 8])
bin_mat = Matrix_test2
global switch_var
switch_var = 1
rows = bin_mat.shape[0]
cols = bin_mat.shape[1]


def scanning(i_start,i_end,j_start,j_end):
    global switch_var 

    for i in range(int(i_start),int(i_end)):            
        for j in range(int(j_start),int(j_end)):
            if bin_mat[i,j] == 1:
                        break
        if bin_mat[i,j] == 1:
            break
        if i==rows:
            switch_var = 0
    return i,j
            
#This function will loop the image until it finds an object
def loop_until_hit(bin_mat,rows,cols,k,i_lower,j_left,j_right):
    global object_matrix

    if k>0:
        # When the bottom of the previous object is in the shadow of the next object       
        if np.sum(bin_mat[int(i_lower),:int(j_left)])>0:
            i_start = object_matrix[k-1,0]
            j_end = object_matrix[k-1,5]     # = j_left[-1]
            [i, j] = scanning(i_start,rows,0,j_end-1)    # Scanning the area to the LEFT of the bottom part of the previous object
      
        # When the bottom of the previous object is casting a shadow over the next object 
        if np.sum(bin_mat[int(i_lower),int(j_right):])>0:
            i_start = object_matrix[k-1,0]
            j_start = object_matrix[k-1,7]
            [i,j] = scanning(i_start,rows,j_start+1,cols) # Scanning the area to the RIGHT of the bottom part of the previous object


    else:
        [i,j] = scanning(0,rows,0,cols)

    object_matrix[k,0] = i
    object_matrix[k,1] = j





#
##SHOULD BE IMPROVED FOR FINDING OBJECTS STACKED ABOVE EACH OTHER AND WITH SAME UPPER HEIGHT (should pass test2 and a test 3 that still has to be made):
#def (Matrix_test, Scaling):
#    global boundary_matrix;
#    global object_matrix;
#
#    #This loop checks for every object wh    cols = len(bin_mat[1,:])
#at it's upper_height is
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
#    return object_matrix
#

#This function finds the right most boundary of an object starting from the coordinates of the most upper point of the object
def right_maxima_finder(Matrix_edges, i, j, k, edge_gap = 0):

    global object_matrix;
    no_land_count=0
    while(no_land_count <= edge_gap):
        if j==cols-1:
            break
        if Matrix_edges[int(i), int(j+1)] == 1:
            j+=1
        elif Matrix_edges[int(i+1), int(j+1)] == 1:
            i+=1
            j+=1
        elif Matrix_edges[int(i+1), int(j)] == 1:
            i+=1
        else:
            #break the while if the edge of matrix is found.
            if j+1==cols-1:
                break
            no_land_count+=1
            j+=1
            if no_land_count>edge_gap:
                j-=edge_gap
    object_matrix[k,2]=i
    object_matrix[k,3]=j

    return j

    

#This function finds the left most edge of the object from the most upper coordinate of the object edge.
def left_maxima_finder(Matrix_edges, i, j, k, edge_gap = 0):

    global object_matrix;
    no_land_count=0
    while(no_land_count <= edge_gap):
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
            #break the while if the edge of matrix is found.
            if j-1==0:
                break
            no_land_count+=1
            j-=1
            if no_land_count>edge_gap:
                j+=edge_gap
    object_matrix[k,4]=i
    object_matrix[k,5]=j

    return j

def lower_maxima_finder(Matrix_edges, i_right, j_right, k_object, edge_gap = 0):
    global object_matrix;
    no_land_count=0
    while(no_land_count <= edge_gap):
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
            #This effectively breaks the while loop, which is intended as there is no use for the edge_gap code anymore
            no_land_count+=1
    object_matrix[k_object, 6]=i_right
    object_matrix[k_object, 7]=j_right




k = 0
while switch_var:
    print('k = ',k)
    if k>0:
        loop_until_hit(bin_mat,rows,cols,k,object_matrix[k-1,6],object_matrix[k-1,5],object_matrix[k-1,3])
    else: 
        loop_until_hit(bin_mat,rows,cols,k,0,0,0)
    right_maxima_finder(bin_mat,object_matrix[k,0],object_matrix[k,1],k)
    left_maxima_finder(bin_mat,object_matrix[k,0],object_matrix[k,1],k)
    lower_maxima_finder(bin_mat,object_matrix[k,2],object_matrix[k,3],k)
    
    
    
    rows_left = int(rows - object_matrix[k,6])
    k += 1
    

object_matrix = object_matrix[~np.all(object_matrix == 0, axis=1)]  # Delete all rows that are only zero 


        


#-----------------------Below here testing the code with the test matrices----------------------------------

#
#plt.figure()
#plt.imshow(Matrix_test1)
#plt.figure()
#plt.imshow(Matrix_test2)
#
#find_upper_extreme(Matrix_test1, Scaling)
#print("Object matrix first object upper_height x coordinate: ", object_matrix[0, 1])
#print("Object matrix second object upper_height y coordinate: ", object_matrix[0,0])
#print(find_upper_extreme(Matrix_test1, Scaling))
##plt.show()
#
#
##edges = [left_edge_finder(Matrix_test1, int(object_matrix[0,0]), int(object_matrix[0,1])) , right_edge_finder(Matrix_test1, int(object_matrix[0,0]), int(object_matrix[0,1]))]
#
#print(Matrix_test2.shape)
#
#
#print("Below here are tests of the maxima finder functions: ")
#find_upper_extreme(Matrix_test1, Scaling)
#right_maxima_finder(Matrix_test1, object_matrix[0, 0], object_matrix[0,1], 0)
#left_maxima_finder(Matrix_test1, object_matrix[0, 0], object_matrix[0,1], 0)
#lower_maxima_finder(Matrix_test1, object_matrix[0, 2], object_matrix[0,3], 0)
#print("The lowest value of the first object is: ", object_matrix[0,6])
#print("The right value of the first object is: ", object_matrix[0,3])
#print("The left value of the first object is: ", object_matrix[0,5])
#print("The highest value of the first object is: ", object_matrix[0,0])
#



                


        



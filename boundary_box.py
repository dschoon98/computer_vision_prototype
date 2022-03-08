import numpy as np

#I will get as an input, an image with ones and zeros indicating, the color edges.
#I have to make a boundary box arround this color edges, so that a more sophisticated edge detector can be ran across that boundary
#Then I either have to make a boundary updater, that re-establishes the boundary around the color edge, or sophisticated edge or simply the same algoritm as before.

#The boundary update should have a scaled boundary box dependent on the object velocity

Scaling = 1;
rows=400
cols=400
object_amount = 10


object_matrix = np.zeros([object_amount, 5])
boundary_matrix = np.zeros([rows,cols])



#This function makes a boundary box arround edges of an object (either color or actual edges)
#Special attention should be paid to multiple objects in the scene
def find_boundary(Matrix_edges, Scaling):
    global boundary_matrix;
    global object_matrix;

    edges_found = 0

    upper_height_temp = 0
    lower_height_temp = 0
    left_width_temp = 0
    right_width_temp = 0

    #This loop checks for every object what it's upper_height is
    #Should break if the object_amount limit is reached
    for i in range(rows):
        edges_row=0
        for j in range(cols):
            if Matrix_edges[i,j]==1:
                edges_row+=1
        
        #check if this returns an integer!
        for k in range(edges_row/2):
            #zero in this case means there is no upper_height assigned or simply not there
            if object_matrix[k,1]!=0:
                object_matrix[k, 1]=i



    #This loop checks for every object what it's lower_height is
    #Should break if the object_amount limit is reached
    a_range = range(rows)
    for i in reversed(a_range):
        edges_row=0
        for j in range(cols):
            if Matrix_edges[i,j]==1:
                edges_row+=1
        
        #check if this returns an integer!
        for k in range(edges_row/2):
            #zero in this case means there is no upper_height assigned or simply not there
            if object_matrix[k,2]!=0:
                object_matrix[k, 2]=i




               


                


        



import numpy as np

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

#In this matrix the different object properties will be stored. In the first collumn is the object ID
#The second collumn depicts the upper_height of the object and the third collumn the lower_height
#the fourth and fifth depict the left and right extreme of the object.
object_matrix = np.zeros([object_amount, 5])



#This will be the matrix the will contain the different boundaries of all the different objects.
#The boundary size away from the limits of the object will be determined by the size of the object (its volume) and the scaling factor.
boundary_matrix = np.zeros([rows,cols])



#This function will find the extremes of the different objects that are in the scene and will display them in the object_matrix
def find_extremes(Matrix_edges, Scaling):
    global boundary_matrix;
    global object_matrix;

    edges_found = 0

    upper_height_temp = 0
    lower_height_temp = 0
    left_width_temp = 0
    right_width_temp = 0

    #This loop checks for every object what it's upper_height is
    #And also is responisble for the the identification of the objects, the highest object will be object 1 and the one below that 2 etc. etc.
    #Should break if the object_amount limit is reached
    for i in range(rows):
        edges_row=0
        #Checks for every row if there are edges present, the edges will always come in pairs
        for j in range(cols):
            if Matrix_edges[i,j]==1:
                edges_row+=1
        #This range below checks how many objects are found in the scene, it does this by using the given that every object has to edges on every row where it lives
        for k in range(int(edges_row/2)):
            #zero in this case means there is no upper_height assigned or simply not there
            #k is in this case revering the objects, thus will write a value to the upper_limmit how k=0 (thus the first object)
            if object_matrix[k,1]==0:
                object_matrix[k, 1]=i



    #This loop checks for every object what it's lower_height is
    #and does this by using a reversed loop
    #The working principle is exactly the same and that is way it incorrectly identifies the correct object order (This algoritm says the lowest object is object 1)
    #This identification issue should be fixed in the future
    a_range = range(rows)
    for i in reversed(a_range):
        edges_row=0
        for j in range(cols):
            if Matrix_edges[i,j]==1:
                edges_row+=1
        
        #check if this returns an integer!
        for k in range(int(edges_row/2)):
            #zero in this case means there is no lower_height assigned or simply not there
            if object_matrix[k,2]==0:
                object_matrix[k, 2]=i


find_extremes(Matrix_test1, Scaling)
#It correctly find the different upper and lower extremes of the two objects only the identifaction is wrong thus the information is stored in the wrong place.
#This will be fixed in the near future.
print(object_matrix[0, 2])


               


                


        


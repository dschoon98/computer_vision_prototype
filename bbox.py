import numpy as np

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


object_amount =10

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
            stopvar_j=0
            for j in range(int(j_start), int(j_end)):
                for K in range(k):
                    if i>=object_matrix[K,0]-1 and i<=object_matrix[K,6]+1:
                        if j>=object_matrix[K,5]:
                            j+=(object_matrix[K,3]-object_matrix[K,5])+1
                            print(j)
                            if j>=j_end-1:
                                stopvar_j=1
                                break
                if stopvar_j:
                    break
                
                if bin_mat[int(i),int(j)]==1:
                    print("HIT!")
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
                #Het programma stopt nu alleen als hij het maximaal aantal objecten heeft gevonden, dit moet anders 
                if k==10:
                    running=0
                break
 
        
        




        
        
        
def object_finder(bin_mat):
    global k
    global object_matrix
    global rows
    global cols
    i_start=0
    i_end=rows
    j_start=0
    j_end=cols
    running=1
    
    #Psuedo code:
    #First the largest object is found in scanning, while scanning the full resolution of the image
    #After which the scanning window is adjusted, so that only the left side of the object is scanned
    #Thus the scan parameters are adjusted
    #The object is incremented so that the correct row of the object_matrix is written
    #The scanning function will indicate whether or not in found a hit
    #If there isn't a hit, it will stop update the search window to look left.
    #This is the tricky part:
    #Now we go back to the first object and go scan right from there we know what the shadowcaster object is
    #,because of the super variable K_SHADOWCASTER.
    #But after the first time it went right, we should use the normal k, because we have written a new object to
    #the current k

    while running: 
        going_left=1
        first_time_right=1
        K_SHADOWCASTER=k
        while going_left:
            scanning(i_start, i_end, j_start, j_end, bin_mat)
            if hit:
                i_start=object_matrix[k,0]
                i_end=object_matrix[k,6]
                j_start=0
                j_end=object_matrix[k, 5]-1
                k+=1
            else:
                going_left=0
        
        while not going_left:
            #Below should only run the first time after going left
            if first_time_right:
                print("First time going right!")
                print("Shadow caster: ", K_SHADOWCASTER)
                i_start=object_matrix[K_SHADOWCASTER,0]
                i_end=object_matrix[K_SHADOWCASTER,6]
                j_start=object_matrix[K_SHADOWCASTER,3]+1
                j_end=cols
                print(i_start, i_end, j_start, j_end)
                scanning(i_start, i_end, j_start, j_end, bin_mat)
                if hit:
                    i_start=object_matrix[k,0]
                    i_end=object_matrix[k,6]
                    j_start=object_matrix[k,3]+1
                    j_end=cols
                    k+=1
                    first_time_right=0
                else:
                    #Should start again the whole sequence but then from the bottom of k_shadowcaster
                    #Thus also update the scan_area
                    i_start=object_matrix[K_SHADOWCASTER,6]
                    i_end=rows
                    j_start=0
                    j_end=cols
                    going_left=1
                    running=0
            else:
                scanning(i_start, i_end, j_start, j_end, bin_mat)
                if hit:
                    i_start=object_matrix[k,0]
                    i_end=object_matrix[k,6]
                    j_start=object_matrix[k,3]+1
                    j_end=cols
                    k+=1
                else:
                    #Should start again from the bottom of the shadowcaster
                    #Update scan area, so that function does the correct thing
                    i_start=object_matrix[K_SHADOWCASTER,6]
                    i_end=rows
                    j_start=0
                    j_end=cols
                    going_left=1
                    running=0


#Where will the whole loop end?:
#We want the programm to stop when the programm has scanned the up until the last row
    
    
    
    
    
    
    
        
        

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

        

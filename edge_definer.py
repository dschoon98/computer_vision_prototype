import numpy as np
import YUV_slices as YUV
import matplotlib.pyplot as plt

resize_factor = 10

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
    return matrix_edge


bin_mat1 = YUV.filter_color('images/flag_and_multi_color.jpeg',180,253,100,150,130,140,resize_factor)
bin_mat2 = YUV.filter_color('images/flag_and_multi_color.jpeg',70,120,150,160,100,120,resize_factor)
bin_mat3 = YUV.filter_color('images/flag_and_multi_color.jpeg',100,200,70,90,160,240,resize_factor)

bin_mat_tot = bin_mat1+bin_mat2+bin_mat3            

matrix_edge = edge_definer(bin_mat_tot)

plt.figure()
plt.imshow(matrix_edge)
plt.title('Edges only')

                

            
    
    





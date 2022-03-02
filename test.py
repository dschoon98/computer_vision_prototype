import numpy as np

matrix = np.zeros([400,400], dtype=object)


for i in range(400):
    for j in range(400):
        temp_array=np.zeros(3)
        for k in range(3):
            temp_array[k]=np.random.rand()

        matrix[i,j]=temp_array


print(matrix)



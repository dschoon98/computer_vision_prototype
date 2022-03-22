import cv2 as cv
import matplotlib.pyplot as plt
import edge_definer as edge 
import numpy as np
import time
resize_factor =1
#img  = cv.imread('images/flag_and_plant.jpeg')
#img = cv.imread('bebop_images/cyberzoo_poles_panels_mats/20190121-142935/49182469.jpg')
#img = cv.imread('bebop_images/cyberzoo_poles_panels_mats/20190121-142935/23616000.jpg')
img = cv.imread('bebop_images/20220304-162806/172689318.jpg')
img = cv.resize(img, (int(img.shape[1]/resize_factor), int(img.shape[0]/resize_factor)));

img = cv.rotate(img,cv.ROTATE_90_COUNTERCLOCKWISE)

img_show = cv.cvtColor(img,cv.COLOR_BGR2RGB)
plt.figure()
plt.imshow(img_show)
#
#for i in range(5):
#    matrix_edges = edge.edge_finder(img,orange=0.1,blue=0.8,white=0.1)
#        
#    plt.figure()
#    plt.imshow(matrix_edges)
#    
start_time = time.time()
#for i in range(5):
#    
#    matrix_edges = edge.edge_finder(img,orange=0.6,blue=0.05,white=0.35)
#    plt.figure()
#    plt.imshow(matrix_edges)
bin_mat = edge.filter_color(img,80,250,130,180,130,180)
plt.figure()
plt.imshow(bin_mat)
end_time = time.time()




print('time = ',end_time-start_time)


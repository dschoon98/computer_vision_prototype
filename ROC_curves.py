import cv2 as cv
import matplotlib.pyplot as plt
import edge_definer as edge 
import numpy as np
import time
import ROC
resize_factor =30
img  = cv.imread('images/flag_and_plant.jpeg')
#img = cv.imread('bebop_images/cyberzoo_poles_panels_mats/20190121-142935/49182469.jpg')
#img = cv.imread('bebop_images/cyberzoo_poles_panels_mats/20190121-142935/23616000.jpg')
#img = cv.imread('bebop_images/20220304-162806/172689318.jpg')
img = cv.resize(img, (int(img.shape[1]/resize_factor), int(img.shape[0]/resize_factor)));

#img = cv.rotate(img,cv.ROTATE_90_COUNTERCLOCKWISE)

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
bin_mat = edge.filter_color(img,80,180,50,100,100,150)
plt.figure()
plt.imshow(bin_mat)
end_time = time.time()






print('time = ',end_time-start_time)
#info = np.iinfo(bin_mat.dtype) # Get the information of the incoming image type
#bin_mat = bin_mat.astype(np.float64) / info.max # normalize the data to 0 - 1
#bin_mat_class = 255 * bin_mat # Now scale by 255
#img_class = bin_mat_class.astype(np.uint8)

bin_mat_class = bin_mat * 255
bin_mat_class = np.dstack((bin_mat_class,bin_mat_class,bin_mat_class))
img_class = bin_mat_class.astype(np.uint8)
RGB, Cl, x, y = ROC.get_images_and_grid(img,img_class)


# Change ONLY the following line:
Values = RGB[:,:,1];

# make it a flat list of values, in the right order:
Values = Values.flatten();

# get the ROC curve:
TP, FP = ROC.get_ROC_curve(Values, Cl);

# plot the ROC curve:
plt.figure();
plt.plot(FP, TP, 'b');
plt.ylabel('TP');
plt.xlabel('FP');

import numpy as np
import cv2 as cv
import bbox
import matplotlib.pyplot as plt


object_amount= 80
# Main script
images_bgr = bbox.load_images_from_folder('CV_output/20_images',binary=False)
images_bin = bbox.load_images_from_folder('CV_output/20_images',binary=True)
plt.figure()
plt.imshow(images_bin[0])
def lukas_kanade(old_bgr,new_bgr,graphics,object_matrix):
    ## parameters - keep them like this:
#     params for ShiTomasi corner detection
#    feature_params = dict( maxCorners = 100,
#                           qualityLevel = 0.3,
#                           minDistance = 7,
#                           blockSize = 7 )
#    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Take first frame and find corners in it
    old_gray = cv.cvtColor(old_bgr, cv.COLOR_BGR2GRAY)
  
#    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
    p0 = np.array([  [object_matrix[0,1],object_matrix[0,0]], 
                     [object_matrix[0,3],object_matrix[0,2]], 
                     [object_matrix[0,5],object_matrix[0,4]], 
                     [object_matrix[0,7],object_matrix[0,6]]     ],np.float32)  # Adding j_right of an obstacle 
    for k in range(object_matrix.shape[0]):
        if k>0:
            added = np.array([ [object_matrix[k,1],object_matrix[k,0]], 
                     [object_matrix[k,3],object_matrix[k,2]], 
                     [object_matrix[k,5],object_matrix[k,4]],
                     [object_matrix[k,7],object_matrix[k,6]]   ],np.float32)
            p0 = np.append(p0,added,axis=0)
        else:
            continue
    
    p0 = p0[:,:,np.newaxis]
    p0 = np.swapaxes(p0,1,2)
    new_gray = cv.cvtColor(new_bgr, cv.COLOR_BGR2GRAY)
    
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)
 
    
#    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    flow_vectors = good_new - good_old
    


    return good_old, good_new, flow_vectors,p0

def determine_optical_flow(images_bin,images_bgr,graphics):
    old_index = 0
    n_images = len(images_bin);

    for im in range(n_images): #n_images
        new_index = old_index + 1
        

        if im>0:

            object_matrix = bbox.x_ray(images_bin[old_index])
            #index right figures and resize them
            old_bgr = images_bgr[old_index]
            new_bgr = images_bgr[new_index]
#            old_bgr = cv.resize(old_bgr, (int(old_bgr.shape[1]/resize_factor), int(old_bgr.shape[0]/resize_factor)));
#            new_bgr = cv.resize(new_bgr, (int(new_bgr.shape[1]/resize_factor), int(new_bgr.shape[0]/resize_factor)));

            # determine optical flow:
            good_old, good_new, flow_vectors,p0 = lukas_kanade(old_bgr,new_bgr,graphics,object_matrix) 

            # convert the pixels to a frame where the coordinate in the center is (0,0)
            good_old -= np.concatenate((0.5*old_bgr.shape[1]*np.ones([good_old.shape[0],1]), 0.5*old_bgr.shape[0]*np.ones([good_old.shape[0],1])),axis=1)
            good_new -= np.concatenate((0.5*old_bgr.shape[1]*np.ones([good_old.shape[0],1]), 0.5*old_bgr.shape[0]*np.ones([good_old.shape[0],1])),axis=1)  # Assumed image size stays the same for each image

            x = good_old[:,[0]]
            y = good_old[:,[1]]
            
            # rotational and translational rates
            A_rot = 0.1  # Unit?
            B_rot = 0.1
            C_rot = 0.1
            U = 1     
            V = 1
            W = 1
            
            u = flow_vectors[:,[0]]
            v = flow_vectors[:,[1]]
            
            # derotate the optical flow vectors 
            u = u - A_rot*np.multiply(x,y) + B_rot*np.multiply(x,x) + B_rot*np.ones([x.shape[0],1]) - C_rot*y
            v = v + C_rot*x - A_rot*np.ones([y.shape[0],1]) - A_rot*np.multiply(y,y) + B_rot*np.multiply(x,y)
            
            Z_horizontal = np.divide(x*W,u) - np.divide(U*np.ones([x.shape[0],1]),u)  # Z = (x*W-U)/u 
            Z_vertical = np.divide(y*W,v) - np.divide(V*np.ones([y.shape[0],1]),v)  # Z = (y*W-V)/v 
            Z = np.abs(np.divide(Z_horizontal + Z_vertical,2*np.ones([x.shape[0],1])))  # Z = (Z_hor + Z_vert)/2
            
            
            old_index += 1
            
            if graphics:
                
                # Move to old coordinate center again, merely for drawing purposes
                good_old += np.concatenate((0.5*old_bgr.shape[1]*np.ones([good_old.shape[0],1]), 0.5*old_bgr.shape[0]*np.ones([good_old.shape[0],1])),axis=1)
                good_new += np.concatenate((0.5*old_bgr.shape[1]*np.ones([good_old.shape[0],1]), 0.5*old_bgr.shape[0]*np.ones([good_old.shape[0],1])),axis=1)  # Assumed image size stays the same for each image
                
                # The image you see is the average of the previous and current image
                ima = (0.5 * old_bgr.copy().astype(float) + 0.5 * new_bgr.copy().astype(float)) / 255.0;
                n_points = good_old.shape[0];
                color = (0,255,0);
                for p in range(n_points):
                    tup_old = tuple(map(int,tuple(good_old[p,:])))
                    tup_new = tuple(map(int,tuple(good_new[p,:])))
                    font = cv.FONT_HERSHEY_SIMPLEX
                    FontScale=0.6
                    thickness = 1
                    cv.arrowedLine(ima, tup_old, tup_new, color,thickness=1,tipLength=0.1);  #Put flow vectors in image
                    #cv.putText(ima,str(round(Z[p][0],2)),(int(good_old[p,0]),int(good_old[p,1])),font,FontScale,color,thickness)  # Put Z_values in image
                cv.imshow('image',ima)
                cv.waitKey(0)    # Value = How many ms each frame stays open
            object_matrix = np.zeros([object_amount,8])
            
    return flow_vectors, good_old, good_new,p0

flow_vectors,good_old,good_new,p0 = determine_optical_flow(images_bin,images_bgr,graphics=True)

cv.destroyAllWindows()


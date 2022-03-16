import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import re
#np.random.seed(328779)
# these functions are to get a nice directory listing
def get_number_file_name(name):
    inds1 = [m.start() for m in re.finditer('_', name)]
    if(inds1 == []):
        return 0;
    ind1 = inds1[-1];
    inds2 = [m.start() for m in re.finditer('\.', name)]
    if(inds2 == []):
        return 0;
    ind2 = inds2[-1];
    number = name[ind1+1:ind2];
    return int(number);

def read_image_folder(image_dir_name,image_type):
    # get the image names from the directory:
    image_names = [];
    for file in os.listdir(image_dir_name):
        if file.endswith(image_type):
            image_names.append(image_dir_name + file);
        image_names.sort(key=get_number_file_name)
    return sorted(image_names)


def lukas_kanade(old_index,new_index,image_names,graphics):
    resize_factor = 2
    ## parameters - keep them like this:
    old_bgr = cv.imread(image_names[old_index])
    new_bgr = cv.imread(image_names[new_index])
    old_bgr = cv.resize(old_bgr, (int(old_bgr.shape[1]/resize_factor), int(old_bgr.shape[0]/resize_factor)));
    new_bgr = cv.resize(new_bgr, (int(new_bgr.shape[1]/resize_factor), int(new_bgr.shape[0]/resize_factor)));

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 40,
                           qualityLevel = 0.1,
                           minDistance = 20,
                           blockSize = 20 )
    
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15, 15),
                      maxLevel = 2,
                      criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
    
    # Take first frame and find corners in it
    old_gray = cv.cvtColor(old_bgr, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    new_gray = cv.cvtColor(new_bgr, cv.COLOR_BGR2GRAY)
    
    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, new_gray, p0, None, **lk_params)

    #cv2.imshow('Flow', im);
    #cv2.waitKey(100);
    #cv2.destroyAllWindows()
    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]

    
    flow_vectors = good_new - good_old
    
    if graphics:

    
        ima = (0.5 * old_bgr.copy().astype(float) + 0.5 * new_bgr.copy().astype(float)) / 255.0;
        n_points = good_old.shape[0];

        color = (0,255,0);
        for p in range(n_points):
            tup_old = tuple(map(int,tuple(good_old[p,:])))
            tup_new = tuple(map(int,tuple(good_new[p,:])))

            cv.arrowedLine(ima, tup_old, tup_new, color,thickness=1,tipLength=0.5);
        
        cv.imshow('image',ima)
        cv.waitKey(0)
        cv.destroyAllWindows()

    return good_old, good_new, flow_vectors
    
#    old_gray = frame_gray.copy()
#    p0 = good_new.reshape(-1, 1, 2)
#    plt.figure()
#    plt.imshow(old_frame)

# Create a mask image for drawing purposes
#mask = np.zeros_like(old_frame)
def ransac(good_old,good_new,flow_vectors,n_iterations,error_threshold,sample_size):
    
    n_points = good_old.shape[0];
     # This is a RANSAC method to better deal with outliers
    # matrices and vectors for the big system:
    A = np.concatenate((good_old, np.ones([good_old.shape[0], 1])), axis=1);
    u_vector = flow_vectors[:,0];
    v_vector = flow_vectors[:,1];
    
    # solve many small systems, calculating the errors:
    errors = np.zeros([n_iterations, 2]);
    pu = np.zeros([n_iterations, 3])
    pv = np.zeros([n_iterations, 3])
    for it in range(n_iterations):
        inds = np.random.choice(range(n_points), size=sample_size, replace=False);
        AA = np.concatenate((good_old[inds,:], np.ones([sample_size, 1])), axis=1);
        pseudo_inverse_AA = np.linalg.pinv(AA);
        # horizontal flow:
        u_vector_small = flow_vectors[inds, 0];
        # pu[it, :] = np.linalg.solve(AA, UU);
        pu[it,:] = np.dot(pseudo_inverse_AA, u_vector_small);
        errs = np.abs(np.dot(A, pu[it,:]) - u_vector);
        errs[errs > error_threshold] = error_threshold;
        errors[it, 0] = np.mean(errs);
        # vertical flow:
        v_vector_small = flow_vectors[inds, 0];
        
        # pv[it, :] = np.linalg.solve(AA, VV);
        pv[it, :] = np.dot(pseudo_inverse_AA, v_vector_small);
        errs = np.abs(np.dot(A, pv[it,:]) - v_vector);
        errs[errs > error_threshold] = error_threshold;

        errors[it, 1] = np.mean(errs);
    
    # take the minimal error
    errors = np.mean(errors, axis=1);
    ind = np.argmin(errors);
    err = errors[ind];
    pu = pu[ind, :];
    pv = pv[ind, :];

    return pu, pv, err

def determine_optical_flow(image_names,graphics):
# iterate over the images:
    old_index = 0
    n_images = len(image_names);
#    FoE_over_time = np.zeros([n_images, 2]);
#    horizontal_motion_over_time = np.zeros([n_images, 1]);
#    vertical_motion_over_time = np.zeros([n_images, 1]);
#    divergence_over_time = np.zeros([n_images, 1]);
#    errors_over_time = np.zeros([n_images, 1]);
##    elapsed_times = np.zeros([n_images,1]);
#    ttc_over_time = np.zeros([n_images,1]);
#    FoE = np.asarray([0.0]*2);
#    time_to_contact = 0.0;
    for im in range(n_images): #n_images
        
        if im>0:
            
    #       t_before = time.time()
            # determine optical flow:
            good_old, good_new, flow_vectors = lukas_kanade(old_index-1,old_index,image_names,graphics) # Right now image indexes are simply 0,1, needs to loop over images depending on output of drone camera
            
    #            elapsed = time.time() - t_before;
    #            if(verbose):
    #                print('Elapsed time = {}'.format(elapsed));
    #            elapsed_times[im] = elapsed;
    #       
            
            # convert the pixels to a frame where the coordinate in the center is (0,0)
            good_old -= 128.0;
            good_new -= 128.0;
            pu, pv, err = ransac(good_old, good_new, flow_vectors,n_iterations=50, error_threshold=10, sample_size=3)
#            print(pu.reshape([3,1]))
#            print(np.array([[good_old[0][0],good_old[0][1],1]]))

            x = good_old[:,[0]]
            y = good_old[:,[1]]
            mat_mul = np.concatenate((x,y,np.ones([x.shape[0],1])),axis=1)
            

            u = np.matmul(mat_mul,pu.reshape([3,1]))
            v = np.matmul(mat_mul,pv.reshape([3,1]))
            print('pu =',pu)
            A_matrix = np.array([       [1,0,-x[0][0],0],
                                        [0,1,-y[0][0],0],
                                        [1,0,-x[28][0],u[28][0]],
                                        [0,1,-y[28][0],u[28][0]] ])
            b_vector = np.array([[u[0][0]],
                                 [v[0][0]],
                                 [0],
                                 [0]])
            solution = np.linalg.solve(A_matrix,b_vector)      
        old_index += 1

            # Solve Ax = b with A being the matrix with 6 equations u1 = ... ,  \n
            # v1 = ..., (...), v3 = ... \
            # b = zeros(6,1), I tried this 
            
            
            # extract the parameters of the flow field:

            
#            # ***********************************
#            # EXERCISE:
#            # change the following lines of code!
#            # ***********************************
#            horizontal_motion = -pu[2];
#            vertical_motion = -pv[2];
#            divergence = (pu[0] + pv[1]) / 2.0;
#            small_threshold = 1E-5;
#            if(abs(pu[0]) > small_threshold):
#                FoE[0] = pu[2] / pu[0]; 
#            if(abs(pv[1]) > small_threshold):
#                FoE[1] = pv[2] / pv[1];    
#            if(abs(divergence) > small_threshold):
#                time_to_contact = 1 / divergence;
#    
#            # book keeping:
#            horizontal_motion_over_time[im] = horizontal_motion;
#            vertical_motion_over_time[im] = vertical_motion;
#            FoE_over_time[im, 0] = FoE[0];
#            FoE_over_time[im, 1] = FoE[1];
#            divergence_over_time[im] = divergence;
#            errors_over_time[im] = err;
#            ttc_over_time[im] = time_to_contact;
    
            # print the FoE and divergence:
#            print('error = {}, FoE = {}, {}, and divergence = {}'.format(err, FoE[0], FoE[1], divergence));
            
    
            # the current image becomes the previous image:

    # ********************************************************************
    # TODO:
    # What is the unit of the divergence?
    # Can you also draw the time-to-contact over time? In what unit is it?
    # ********************************************************************
#    if graphics:
#        
#        plt.figure();
#        plt.plot(range(n_images), divergence_over_time, label='Divergence');
#        plt.xlabel('Image')
#        plt.ylabel('Divergence')
#    
#        plt.figure();
#        plt.plot(range(n_images), FoE_over_time[:,0], label='FoE[0]');
#        plt.plot(range(n_images), FoE_over_time[:,1], label='FoE[1]');
#        plt.plot(range(n_images), np.asarray([0.0]*n_images), label='Center of the image')
#        plt.legend();
#        plt.xlabel('Image')
#        plt.ylabel('FoE')
#    
#        plt.figure();
#        plt.plot(range(n_images), errors_over_time, label='Error');
#        plt.xlabel('Image')
#        plt.ylabel('Error')
#    
#        plt.figure();
#        plt.plot(range(n_images), horizontal_motion_over_time, label='Horizontal motion');
#        plt.plot(range(n_images), vertical_motion_over_time, label='Vertical motion');
#        plt.legend();
#        plt.xlabel('Image')
#        plt.ylabel('Motion U/Z, V/Z')       
#    
#        plt.figure();
#        plt.plot(range(n_images), ttc_over_time, label='Time-to-contact');
#        plt.xlabel('Image')
#        plt.ylabel('Time-to-contact')

    return flow_vectors, good_old, good_new, solution

# Main script
image_names = read_image_folder('images/opticflow/','jpeg')
flow_vectors,good_old,good_new,solution = determine_optical_flow(image_names,graphics=False)



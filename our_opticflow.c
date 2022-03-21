#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "pthread.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "opencv_image_functions.h"
#include "opencv_example.h"

int object_amount = 80;
void lukas_kanade(int old_bgr, int new_bgr, int graphics, float object_matrix) 
{
    float p0[4][2] = {
      {object_matrix[0][1],object_matrix[0][0],object_matrix[0][3],(object_matrix[0][0]-(object_matrix[0][6]/2)),object_matrix[0,5],(object_matrix[0][0]-(object_matrix[0][6]/2)),object_matrix[0][7],object_matrix[0][6]}
    }
    float added[4][2] = {
      {object_matrix[k][1],object_matrix[k][0],object_matrix[k][3],(object_matrix[0][0]-(object_matrix[0][6]/2)),object_matrix[k,5],(object_matrix[0][0]-(object_matrix[0][6]/2)),object_matrix[k][7],object_matrix[k][6]}
    }

    for(int k = 0, k<object_amount, k++) {
      if(k>0) {
        float added[4][2] = {};
        added[0][0] = object_matrix[k][1]
        added = 
      }
    }

}

void determine_optical_flow(int images_bin, int images_bgr, int graphics) 
{
    int old_index = 0;
    int n_images = sizeof(images_bin);
    
    // rate definitions //
    float A_rot = 0.1;
    float B_rot = 0.1;
    float C_rot = 0.1;
    int U = 1;
    int V = 1;
    int W = 1;
    // rate definitions //




}
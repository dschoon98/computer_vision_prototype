#include <stdio.h>
#include <stdbool.h>
#include <math.h>
#include "pthread.h"

int test_matrix[22][37] = {
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,0,0,0,0,0,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,1,1,1,1,1,1,1,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0},
    };

int k_object = 0;
int save_i = 0; int save_j = 0;
int object_matrix[10][8] = {
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
  {0,0,0,0,0,0,0,0},
};

const int rows = 22;
const int cols = 37;

void right_maxima_finder(int bin_mat[rows][cols],int i, int j, int* k_object, int object_matrix[10][8])
{   
    int local_i = i; int local_j = j;
    bool running = true;
    bool running2 = true;
    while(running) {
      while(running2) {
        if(j == cols-1) {
          running=false;
        } 
        else if(bin_mat[local_i][local_j+1] == 1) {
          local_i++;
          running=false;
        }
        else if(bin_mat[local_i+1][local_j+1] == 1) {
          local_i++;
          local_j++;
          running=false;
        }
        else if(bin_mat[local_i+1][local_j] == 1) {
          local_i++;
          running=false;
        }
        else {
          running2=false;
        }
      }
    }
    object_matrix[*k_object][2] = local_i;
    object_matrix[*k_object][3] = local_j;
}

void left_maxima_finder(int bin_mat[rows][cols],int i, int j, int* k_object, int object_matrix[10][8])
{   
    int local_i = i; int local_j = j;
    bool running = true;
    bool running2 = true;
    while(running) {
      while(running2) {
        if(j == cols-1) {
          running=false;
        } 
        else if(bin_mat[local_i][local_j-1] == 1) {
          local_j--;
          running=false;
        }
        else if(bin_mat[local_i+1][local_j-1] == 1) {
          local_i++;
          local_j--;
          running=false;
        }
        else if(bin_mat[local_i+1][local_j] == 1) {
          local_i++;
          running=false;
        }
        else {
          running2=false;
        }
      }
    }
    object_matrix[*k_object][4] = local_i;
    object_matrix[*k_object][5] = local_j;
}

void lower_maxima_finder(int bin_mat[rows][cols],int i, int j, int* k_object, int object_matrix[10][8])
{   
    int local_i = i; int local_j = j;
    bool running = true;
    bool running2 = true;
    while(running) {
      while(running2) {
        if(j == rows-1) {
          running=false;
        } 
        else if(bin_mat[local_i+1][local_j] == 1) {
          local_i++;
          running=false;
        }
        else if(bin_mat[local_i+1][local_j-1] == 1) {
          local_i++;
          local_j--;
          running=false;
        }
        else if(bin_mat[local_i][local_j-1] == 1) {
          local_j--;
          running=false;
        }
        else {
          running2=false;
        }
      }
    }
    object_matrix[*k_object][6] = local_i;
    object_matrix[*k_object][7] = local_j;
}

int main(){
  right_maxima_finder(test_matrix,3,9,&k_object,object_matrix);
  left_maxima_finder(test_matrix,3,9,&k_object,object_matrix);
  lower_maxima_finder(test_matrix,3,9,&k_object,object_matrix);
  //x_ray(test_matrix,&k_object,object_matrix);
  printf("\nk = %d", k_object);

  printf("\nThe Array elements are:\n");
  // outer loop for row
  for(int i=0; i<10; i++) {
    // inner loop for column
    for(int j=0; j<8; j++) {
      printf("%d ", object_matrix[i][j]);
    }
    printf("\n"); // new line
  }

  return 0;
} 
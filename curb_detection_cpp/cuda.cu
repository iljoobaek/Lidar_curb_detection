#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdbool.h>
#include <math.h>






__global__ void cont_filter (int st, float* dist_to_origin, float *pc,
                                 bool* is_continuous,  bool* this_continuous )
{   
    const int i = threadIdx.x;
    float THETA_R = 0.00356999;
    float thres = dist_to_origin[st+i] * THETA_R * 7;
    int j = st+i;

    float val = sqrt((pc[7*(j+1)]-pc[7*j])*(pc[7*(j+1)]-pc[7*j]) 
                    + (pc[7*(j+1)+1]-pc[7*j+1])*(pc[7*(j+1)+1]-pc[7*j+1]) 
                    + (pc[7*(j+1)+2]-pc[7*j+2])*(pc[7*(j+1)+2]-pc[7*j+2]));
    
    if (val > thres)
    {
        is_continuous[i] = false;
        is_continuous[i+1] = false;
        this_continuous[st+i] = false;
        this_continuous[st+i+1] = false;
    }

}

void continuous_filter_gpu2 (int n, float* dist_to_origin )
{
    
}

void continuous_filter_gpu2 (int n, int st, float* dist_to_origin, float *pc,
                                 bool* is_continuous,  bool* this_continuous )
{
    int num_threads= 512;
    int num_blocks = int((n+num_threads-1)/num_threads);
    cont_filter <<< num_threads, num_blocks >>> ( st, dist_to_origin,pc,is_continuous, this_continuous);
    cudaDeviceSynchronize();
}














// __global__ void elevation filter()
// {
//     float thres_z = 0.005;
//     const int i = threadIdx.x;

//     if (scan_id % 2 ==0)
//     {
//         if (i< n-2)
//         {
//             z_diff[i] = pointcloud[st+i][2] - pointcloud[st+i+1][2];
//         }

//     }
//     else
//     {
//         if (i>0)
//         {
//             z_diff[i] = pointcloud[st+i][2] - pointcloud[st+i-1][2];
//         }
//     }

//     if (z_diff[i] > thres_z)
//     {
//         is_elevate[i] = 1;
//     }

//     __syncthreads();

//     conv <<< 1,256 >>> (elevation, f_start, edge_start_cnt);
//     cudaDeviceSynchronize();

//     __syncthreads();

//     if out[i+4] >= 4
//     {
//         is_elevate[i] = 1;
//         is_elevating[i] = true;
//     }
//     else
//     {
//         is_elevate[i] = 0;
//         is_elevating[i] = true;
//     }

// }


// __global__ void edge_filter_from_elevation()
// {
//     const int i = threadIdx.x;
//     int k = 7;

//     __syncthreads();

//     conv <<< 1,256 >>> (elevation, f_start, edge_start_cnt);
//     conv <<< 1,256 >>> (elevation, f_end, edge_end_cnt);

//     cudaDeviceSynchronize();

//     __syncthreads();
    

//     if (edge_start_cnt[i+k] >= 2 )
//     {
//         edge_start[i] = true;
//         is_edge_start[st+i] = true;
//     }
//     if (edge_end_cnt[i+k] >= 2 )
//     {
//         edge_end[i] = true;
//         is_edge_end[st+i] = true;
//     }

// }


// __global__ void get_local_min()
// {
//     const int i = threadIdx.x;
//     diff = vec[i+1] - vec[i];

//     if (diff > 0.0f)
//         first_derivative[i] = 1;
//     else if (diff < 0.0f)
//         first_derivative[i] = -1;
//     else
//         first_derivative[i] = 0;
    
//     diff = first_derivative[i+1] - first_derivative[i];

//     if (diff > 0)
//         second_derivative[i] = true;

// }


// __global__ void local_min_of_direction_change()
// {
//     const int i = threadIdx.x;

//     if (direction[i] < 150.0f)
//     {
//         direction_change[i] = true;
//     }

//     __syncthreads();

//     local_min <<< 1,256 >>> (direction);
//     cudaDeviceSynchronize();

//     __syncthreads();

//     direction_change_local_min[i] = direction_change[i] && local_min[i];
//     is_local_min[st+i] = directional_change[i] && local_min[i];

// }













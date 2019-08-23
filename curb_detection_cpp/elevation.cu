#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>


thrust::host_vector<int> Boundary_detection::elevation_filter_gpu(int scan_id) {
    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
    int n = ed - st;
    thrust::host_vector<int> is_elevate(n, 0);
    thrust::host_vector<float> z_diff(n, 0.0f); 
    float thres_z = 0.005;

    if (scan_id % 2 == 0) { // left scan
        for (int i = n-2; i >= 0; i--) {
            z_diff[i] = this->pointcloud[st+i][2] - this->pointcloud[st+i+1][2];
        }
    }
    else {
        for (int i = 1; i < n; i++) {
            z_diff[i] = this->pointcloud[st+i][2] - this->pointcloud[st+i-1][2];
        }
    }
    for (int i = 0; i < n; i++) {
        if (z_diff[i] > thres_z) is_elevate[i] = 1;
    }
    thrust::host_vector<int> filter({1,1,1,1,1,1,1,1,1});
    auto res = conv(is_elevate, filter);
    for (int i = 0; i < is_elevate.size(); i++) {
        if (res[i+4] >= 4) is_elevate[i] = 1;
        else is_elevate[i] = 0;
    }
    for (int i = 0; i < is_elevate.size(); i++) {
        if (is_elevate[i] > 0) this->is_elevating[st+i] = true;
    }
    return is_elevate;
}
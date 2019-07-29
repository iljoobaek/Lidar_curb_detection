// #include <opencv2/opencv.hpp>
// #include <opencv2/viz.hpp>
// #include <Eigen/Dense>

#include "boundary_detection.h"
// Include VelodyneCapture Header
// #include "VelodyneCapture.h"

int main( int argc, char* argv[] ) {
    cout << "Test Eigen and c++ computation" << endl; 
    string filename = "image.bin";
    // vector<vector<float>> pointcloud = read_bin(filename);
    Boundary_detection *detection = new Boundary_detection(filename);

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    
    vector<float> dist_to_origin = detection->get_dist_to_origin(detection->get_pointcloud());
    vector<bool> is_continuous = detection->continuous_filter(detection->get_pointcloud(), dist_to_origin);
    detection->print_pointcloud(detection->get_pointcloud());
    
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(t2 - t1).count();
    cout << duration << endl;

    return 0;
}

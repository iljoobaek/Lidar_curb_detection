// #include <opencv2/opencv.hpp>
// #include <opencv2/viz.hpp>

#include "boundary_detection.h"
// Include VelodyneCapture Header
// #include "VelodyneCapture.h"

int main( int argc, char* argv[] ) {
    cout << "Test Eigen and c++ computation" << endl; 

    string filename = "image.bin";
    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    Boundary_detection *detection = new Boundary_detection(filename, 0, 15.0, 1.125);
    vector<bool> result = detection->run_detection();    
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(t2 - t1).count();
    cout << duration << endl;
    //detection->print_pointcloud(detection->get_pointcloud());
    
    // string folder = "test/";
    // for (int i = 0; i < 1171; i++) {
    // // for (int i = 250; i < 251; i++) {
    //     cout << "frame: " << i << endl;
    //     string filename = folder + std::to_string(i) + ".bin";
    //     Boundary_detection *detection = new Boundary_detection(filename, i, 15.0, 1.125);

    //     high_resolution_clock::time_point t1 = high_resolution_clock::now();
        
    //     vector<float> dist_to_origin = detection->get_dist_to_origin(detection->get_pointcloud());
    //     vector<bool> is_continuous = detection->continuous_filter(detection->get_pointcloud(), dist_to_origin);
        
    //     high_resolution_clock::time_point t2 = high_resolution_clock::now();
    //     auto duration = duration_cast<microseconds>(t2 - t1).count();
        
    //     // detection->print_pointcloud(detection->get_pointcloud());
        
    //     high_resolution_clock::time_point t3 = high_resolution_clock::now();
    //     auto duration2 = duration_cast<microseconds>(t3 - t2).count();
        
    //     // cout << duration << endl;
    //     // cout << duration2 << endl;

    // }

    return 0;
}

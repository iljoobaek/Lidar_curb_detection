#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <stack>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>

#include <Eigen/Dense>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include "VelodyneCapture.h"

#include "Fusion.cpp"

#define PI 3.14159265
#define THETA_R 0.00356999

using namespace std::chrono;

typedef boost::interprocess::allocator<cv::Vec3f, boost::interprocess::managed_shared_memory::segment_manager>  ShmemAllocator;
typedef boost::interprocess::vector<cv::Vec3f, ShmemAllocator> radar_shared;

class Boundary_detection 
{
private:
    enum class ScanDirection 
    {
        CLOCKWISE,
        COUNTER_CLOCKWISE
    };
public:
    Boundary_detection(std::string dir, float tilted_angle, float sensor_height): 
                        directory(dir), num_of_scan(16), 
                        tilted_angle(tilted_angle), sensor_height(sensor_height) 
    {
        ranges = std::vector<std::vector<int>>(16, std::vector<int>(2));
        angles = {-15.0, 1.0, -13.0, 3.0, -11.0, 5.0, -9.0, 7.0,
                        -7.0, 9.0, -5.0, 11.0, -3.0, 13.0, -1.0, 15.0};
        //timedFunction(std::bind(&Boundary_detection::expose, this), 100);
        if (dir.find(".pcap") != std::string::npos) {
            isPCAP = true;
            capture = std::unique_ptr<velodyne::VLP16Capture> (new velodyne::VLP16Capture(dir));
        }
        else {
            isPCAP = false;
        }
        fuser = fusion::FusionController();
    } 
    
    void rotate_and_translate_multi_lidar_yaw(const cv::Mat &rot);
    void max_height_filter(float max_height);
    void rearrange_pointcloud();
    void pointcloud_preprocessing(const cv::Mat &rot);

    void detect(const cv::Mat &rot, const cv::Mat &trans, bool vis=false);

    void reset();    
    std::vector<std::vector<float>>& get_pointcloud();
    std::vector<int>& get_result();
    std::vector<bool>& get_result_bool();

    std::vector<std::vector<cv::Vec3f>> getLidarBuffers(const std::vector<std::vector<float>> &pointcloud, const std::vector<bool> &result);
    void timedFunction(std::function<void(void)> func, unsigned int interval);
    void expose();
    
    boost::interprocess::named_mutex 
            mem_mutex{
                boost::interprocess::open_or_create, 
                "radar_mutex"
            };
    
    std::unique_ptr<velodyne::VLP16Capture> capture;

private:
    bool isPCAP;
    bool firstRun = true;
    bool secondRun = false;
    fusion::FusionController fuser;
    std::string directory;
    int num_of_scan;
    float tilted_angle;
    float sensor_height;
    std::vector<float> angles;
    std::vector<std::vector<float>> pointcloud;
    std::vector<bool> is_boundary;
    std::vector<int> is_boundary_int;
    std::vector<std::vector<int>> ranges;
    std::vector<cv::Vec3f> radar_pointcloud;
};

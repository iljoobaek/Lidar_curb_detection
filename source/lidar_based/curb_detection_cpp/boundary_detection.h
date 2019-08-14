#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <thread>

#include <Eigen/Dense>

#include "VelodyneCapture.h"
#include "Fusion.cpp"
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>

#define PI 3.14159265
#define THETA_R 0.00356999
#define MIN_CURB_HEIGHT 0.05

using std::cout;
using std::endl;
using std::vector;
using std::string;
using namespace std::chrono;

typedef boost::interprocess::allocator<cv::Vec3f, boost::interprocess::managed_shared_memory::segment_manager>  ShmemAllocator;
typedef boost::interprocess::vector<cv::Vec3f, ShmemAllocator> radar_shared;

class Boundary_detection {
public:
    Boundary_detection(string dir, int id, float tilted_angle, float sensor_height): directory(dir), frame_id(id), num_of_scan(16) {
        this->ranges = std::vector<std::vector<int>>(32, std::vector<int>(2));
        this->tilted_angle = tilted_angle;
        this->sensor_height = sensor_height;
        this->angles = {-15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0,
                        1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0};
        //timedFunction(std::bind(&Boundary_detection::expose, this), 100);
        if (dir.find(".pcap") != string::npos) this->isPCAP = true;
        else this->isPCAP = false;
        this->fuser = fusion::FusionController();
    } 

    void laser_to_cartesian(std::vector<velodyne::Laser>& lasers);
    std::vector<std::vector<float>> read_bin(string filename);
    void rotate_and_translate();
    void max_height_filter(float max_height);
    void reorder_pointcloud();
    void rearrange_pointcloud();
    void rearrange_pointcloud_sort();
    void pointcloud_preprocessing();
    
    float dist_between(const std::vector<float>& p1, const std::vector<float>& p2);
    std::vector<float> get_dist_to_origin();
    std::vector<float> get_theoretical_dist();
    std::vector<bool> continuous_filter(int scan_id);
    float get_angle(const std::vector<float>& v1, const std::vector<float>& v2);
    std::vector<float> direction_change_filter(int scan_id, int k, float angle_thres=150.0f);
    std::vector<bool> local_min_of_direction_change(int scan_id);
    std::vector<int> elevation_filter(int scan_id);
    void edge_filter_from_elevation(int scan_id, const std::vector<int>& elevation, std::vector<bool>& edge_start, std::vector<bool>& edge_end);
    std::vector<bool> obstacle_extraction(int scan_id);

    void find_boundary_from_half_scan(int scan_id, int k);
    std::vector<bool> run_detection(bool vis=false);

    void print_pointcloud(const std::vector<std::vector<float>>& pointcloud);

    void reset();    
    std::vector<std::vector<float>>& get_pointcloud();
    std::vector<bool>& get_result();

    std::vector<std::vector<cv::Vec3f>> getLidarBuffers(const std::vector<std::vector<float>>& pointcloud, const std::vector<bool>& result);
    void timedFunction(std::function<void(void)> func, unsigned int interval);
    void expose();
    
    boost::interprocess::named_mutex 
            mem_mutex{
                boost::interprocess::open_or_create, 
                "radar_mutex"
            };

private:
    bool isPCAP;
    bool firstRun = true;
    fusion::FusionController fuser;
    string directory;
    int frame_id;
    int num_of_scan;
    float tilted_angle;
    float sensor_height;
    std::vector<float> angles;
    std::vector<float> dist_to_origin;
    std::vector<float> theoretical_dist;
    std::vector<std::vector<float>> pointcloud;
    std::vector<bool> is_boundary;
    std::vector<bool> is_continuous;
    std::vector<bool> is_elevating;
    std::vector<bool> is_changing_angle;
    std::vector<bool> is_local_min;
    std::vector<bool> is_edge_start;
    std::vector<bool> is_edge_end;
    std::vector<bool> is_obstacle;
    std::vector<std::vector<int>> ranges;
    std::vector<cv::Vec3f> radar_pointcloud;
};

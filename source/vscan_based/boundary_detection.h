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

#ifndef CV2_H
#define CV2_H
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#endif

#include "data_reader.h"
#include "Fusion.cpp"

#define PI 3.14159265
#define THETA_R 0.00356999
#define MIN_CURB_HEIGHT 0.05

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
    Boundary_detection(float tilted_angle, float sensor_height, std::string data_path, int start, int end): 
                        num_of_scan(16), dataReader(data_path, start, end), 
                        tilted_angle(tilted_angle), sensor_height(sensor_height) 
    {
        ranges = std::vector<std::vector<int>>(32, std::vector<int>(2));
        angles = {-15.0, 1.0, -13.0, 3.0, -11.0, 5.0, -9.0, 7.0,
                        -7.0, 9.0, -5.0, 11.0, -3.0, 13.0, -1.0, 15.0};
        //timedFunction(std::bind(&Boundary_detection::expose, this), 100);
        fuser = fusion::FusionController();
    } 
    Boundary_detection(float tilted_angle, float sensor_height, std::string data_path): 
                        num_of_scan(16), dataReader(data_path), 
                        tilted_angle(tilted_angle), sensor_height(sensor_height) 
    {
        ranges = std::vector<std::vector<int>>(32, std::vector<int>(2));
        angles = {-15.0, 1.0, -13.0, 3.0, -11.0, 5.0, -9.0, 7.0,
                        -7.0, 9.0, -5.0, 11.0, -3.0, 13.0, -1.0, 15.0};
        //timedFunction(std::bind(&Boundary_detection::expose, this), 100);
        fuser = fusion::FusionController();
    }

    bool isRun();
    void retrieveData();
    void pointcloud_preprocessing(const cv::Mat &rot);
    void detect(const cv::Mat &rot, const cv::Mat &trans);
    std::vector<std::vector<float>>& get_pointcloud();
    std::vector<int> get_result();
    std::vector<bool> get_result_bool();
    std::vector<std::vector<cv::Vec3f>> getLidarBuffers(const std::vector<std::vector<float>> &pointcloud, const std::vector<bool> &result);
    std::vector<cv::viz::WPolyLine> getThirdOrderLines(std::vector<cv::Vec3f> &buf); 

private:
    void rotate_and_translate_multi_lidar_yaw(const cv::Mat &rot);
    void max_height_filter(float max_height);
    void rearrange_pointcloud();

    std::vector<float> get_dist_to_origin();
    float dist_between(const std::vector<float> &p1, const std::vector<float> &p2);
    std::vector<bool> continuous_filter(int scan_id);
    float get_angle(const std::vector<float> &v1, const std::vector<float> &v2);
    std::vector<float> direction_change_filter(int scan_id, int k, float angle_thres=150.0f);
    std::vector<bool> local_min_of_direction_change(int scan_id);
    std::vector<int> elevation_filter(int scan_id);
    void edge_filter_from_elevation(int scan_id, const std::vector<int> &elevation, std::vector<bool> &edge_start, std::vector<bool> &edge_end);
    float distance_to_line(cv::Point2f p1, cv::Point2f p2);

    void find_boundary_from_half_scan(int scan_id, int k, bool masking);

    void reset();    

    void timedFunction(std::function<void(void)> func, unsigned int interval);
    void expose();
    
    boost::interprocess::named_mutex 
            mem_mutex{
                boost::interprocess::open_or_create, 
                "radar_mutex"
            };
    
private:
    bool firstRun = true;
    bool secondRun = false;
    fusion::FusionController fuser;

    DataReader::LidarDataReader dataReader;

    int num_of_scan;
    float tilted_angle;
    float sensor_height;
    std::vector<float> angles;
    
    std::vector<std::vector<float>> pointcloud;
    std::vector<std::vector<int>> ranges;
    std::vector<cv::Vec3f> radar_pointcloud;
    std::vector<float> dist_to_origin;

    std::vector<bool> is_boundary;
    std::vector<int> is_boundary_int;

    std::vector<bool> is_continuous;
    std::vector<bool> is_elevating;
    std::vector<bool> is_changing_angle;
    std::vector<bool> is_local_min;
    std::vector<bool> is_edge_start;
    std::vector<bool> is_edge_end;

    std::vector<bool> is_boundary_masking;
    std::vector<bool> is_objects;
};

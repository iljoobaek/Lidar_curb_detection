#include <vector>
#include <opencv2/opencv.hpp>

namespace SensorConfig
{
std::vector<std::string> getPcapFiles() 
{
    std::vector<std::string> files(6);
    // files[0] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_2/mil19_road_2019.9.13_center_front_2.pcap";
    // files[1] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_2/mil19_road_2019.9.13_center_rear_2.pcap";
    // files[2] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_2/mil19_road_2019.9.13_left_front_2.pcap";
    // files[3] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_2/mil19_road_2019.9.13_left_side_2.pcap";
    // files[4] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_2/mil19_road_2019.9.13_right_front_2.pcap";
    // files[5] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_2/mil19_road_2019.9.13_right_side_2.pcap";

    // files[0] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_5/mil19_road_2019.9.13_center_front_5.pcap";
    // files[1] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_5/mil19_road_2019.9.13_center_rear_5.pcap";
    // files[2] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_5/mil19_road_2019.9.13_left_front_5.pcap";
    // files[3] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_5/mil19_road_2019.9.13_left_side_5.pcap";
    // files[4] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_5/mil19_road_2019.9.13_right_front_5.pcap";
    // files[5] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_5/mil19_road_2019.9.13_right_side_5.pcap";
    
    files[0] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_2/mil19_indoor_2_front_center.pcap";
    files[1] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_2/mil19_indoor_2_rear_center.pcap";
    files[2] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_2/mil19_indoor_2_front_left.pcap";
    files[3] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_2/mil19_indoor_2_side_left.pcap";
    files[4] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_2/mil19_indoor_2_front_right.pcap";
    files[5] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_2/mil19_indoor_2_side_right.pcap";
    
    // files[0] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_1/mil19_indoor_1_front_center.pcap";
    // files[1] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_1/mil19_indoor_1_rear_center.pcap";
    // files[2] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_1/mil19_indoor_1_front_left.pcap";
    // files[3] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_1/mil19_indoor_1_side_left.pcap";
    // files[4] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_1/mil19_indoor_1_front_right.pcap";
    // files[5] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_1/mil19_indoor_1_side_right.pcap";
    return files;
}

std::vector<float> getRotationParams() 
{
    std::vector<float> rot_params;
    rot_params.push_back(-14.0f);  // center front
    rot_params.push_back(-119.0f);  // center rear
    rot_params.push_back(-44.0f);  // driver front (left front)
    rot_params.push_back(-125.0f); // driver side (left side)
    rot_params.push_back(38.5f);   // passenger front (right front)
    rot_params.push_back(40.0f);  // passenger side (right side)
    return rot_params;
}

std::vector<cv::Mat> getRotationMatrices(const std::vector<float> &rot_params) 
{
    // rotation matrix along z axis 
    std::vector<cv::Mat> rotation_matrices;
    for (int i = 0; i < rot_params.size(); i++) 
    {
        cv::Mat rot = cv::Mat::zeros(3, 3, CV_32FC1);
        float theta = -rot_params[i] * CV_PI / 180.;  // Notice the sign of theta!
        rot.at<float>(0,0) = std::cos(theta); 
        rot.at<float>(1,0) = std::sin(theta); 
        rot.at<float>(2,0) = 0.0f;
        rot.at<float>(0,1) = -std::sin(theta); 
        rot.at<float>(1,1) = std::cos(theta); 
        rot.at<float>(2,2) = 0.0f;
        rot.at<float>(0,2) = 0.0f; 
        rot.at<float>(1,2) = 0.0f; 
        rot.at<float>(2,2) = 1.0f;
        rotation_matrices.push_back(rot); 
    }
    return rotation_matrices;
}

std::vector<cv::Mat> getTranslationMatrices() {
    std::vector<std::vector<float>> trans_params;
    trans_params.push_back({3.98f, 0.00f, -0.22f});  // center front
    trans_params.push_back({-1.19f, 0.00f, 0.14f});  // center rear
    trans_params.push_back({3.91f, -0.60f, 0.00f});  // driver front (left front)
    trans_params.push_back({2.70f, -0.90f, -0.55f}); // driver side (left side)
    trans_params.push_back({3.83f, 0.61f, 0.00f});   // passenger front (right front)
    trans_params.push_back({2.70f, 0.90f, -0.50f});  // passenger side (right side)

    std::vector<cv::Mat> translation_matrices;
    for (int i = 0; i < trans_params.size(); i++) 
    {
        cv::Mat trans = cv::Mat::zeros(3, 1, CV_32FC1);
        trans.at<float>(0,0) = trans_params[i][0]; 
        trans.at<float>(1,0) = trans_params[i][1]; 
        trans.at<float>(2,0) = trans_params[i][2];
        translation_matrices.push_back(trans); 
    }
    return translation_matrices;
}
}
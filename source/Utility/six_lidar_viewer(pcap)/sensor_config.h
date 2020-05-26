#include <vector>
#include <string>
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
    
    files[0] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_5/mil19_road_2019.9.13_center_front_5.pcap";
    files[1] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_5/mil19_road_2019.9.13_center_rear_5.pcap";
    files[2] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_5/mil19_road_2019.9.13_left_front_5.pcap";
    files[3] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_5/mil19_road_2019.9.13_left_side_5.pcap";
    files[4] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_5/mil19_road_2019.9.13_right_front_5.pcap";
    files[5] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_road_2019.9.13_5/mil19_road_2019.9.13_right_side_5.pcap";
    
    // files[0] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_2/mil19_indoor_2_front_center.pcap";
    // files[1] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_2/mil19_indoor_2_rear_center.pcap";
    // files[2] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_2/mil19_indoor_2_front_left.pcap";
    // files[3] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_2/mil19_indoor_2_side_left.pcap";
    // files[4] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_2/mil19_indoor_2_front_right.pcap";
    // files[5] = "/home/rtml/LiDAR_camera_calibration_work/data/mil19_indoor_2/mil19_indoor_2_side_right.pcap";
    return files;
}

std::string getBinaryFile(int frame_idx, std::string root_dir="autoware-20190828124709/")
{
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(10) << frame_idx;
    std::string filename = "/home/rtml/LiDAR_camera_calibration_work/data/data_raw/synced/" + root_dir + "velodyne_points/data/" + ss.str() + ".bin";
    return filename;
}

std::vector<float> getRotationParams() 
{
    std::vector<float> rot_params;
    rot_params.push_back(-14.0f);  // center front
    rot_params.push_back(-119.0f);  // center rear
    // rot_params.push_back(-44.0f);  // driver front (left front)
    rot_params.push_back(-43.65f);  // driver front (left front)
    rot_params.push_back(-125.0f); // driver side (left side)
    rot_params.push_back(38.5f);   // passenger front (right front)
    rot_params.push_back(40.0f);  // passenger side (right side)
    return rot_params;
}

cv::Mat getRotationMatrixFromTheta_X(float theta_deg)
{
    cv::Mat rot = cv::Mat::zeros(3, 3, CV_32FC1);
    float theta_rad = -theta_deg * CV_PI / 180.;  // Notice the sign of theta!
    rot.at<float>(0,0) = 1.0f;
    rot.at<float>(1,0) = 0.0f;
    rot.at<float>(2,0) = 0.0f; 
    rot.at<float>(0,1) = 0.0f;
    rot.at<float>(1,1) = std::cos(theta_rad); 
    rot.at<float>(2,1) = std::sin(theta_rad); 
    rot.at<float>(0,2) = 0.0f;
    rot.at<float>(1,2) = -std::sin(theta_rad); 
    rot.at<float>(2,2) = std::cos(theta_rad); 
    return rot;
}

cv::Mat getRotationMatrixFromTheta_Y(float theta_deg)
{
    cv::Mat rot = cv::Mat::zeros(3, 3, CV_32FC1);
    float theta_rad = -theta_deg * CV_PI / 180.;  // Notice the sign of theta!
    rot.at<float>(0,0) = std::cos(theta_rad); 
    rot.at<float>(1,0) = 0.0f; 
    rot.at<float>(2,0) = -std::sin(theta_rad);
    rot.at<float>(0,1) = 0.0f; 
    rot.at<float>(1,1) = 1.0f; 
    rot.at<float>(2,1) = 0.0f;
    rot.at<float>(0,2) = std::sin(theta_rad); 
    rot.at<float>(1,2) = 0.0f; 
    rot.at<float>(2,2) = std::cos(theta_rad);
    return rot;
}

cv::Mat getRotationMatrixFromTheta_Z(float theta_deg)
{
    cv::Mat rot = cv::Mat::zeros(3, 3, CV_32FC1);
    float theta_rad = -theta_deg * CV_PI / 180.;  // Notice the sign of theta!
    rot.at<float>(0,0) = std::cos(theta_rad); 
    rot.at<float>(1,0) = std::sin(theta_rad); 
    rot.at<float>(2,0) = 0.0f;
    rot.at<float>(0,1) = -std::sin(theta_rad); 
    rot.at<float>(1,1) = std::cos(theta_rad); 
    rot.at<float>(2,1) = 0.0f;
    rot.at<float>(0,2) = 0.0f; 
    rot.at<float>(1,2) = 0.0f; 
    rot.at<float>(2,2) = 1.0f;
    return rot;
}

cv::Mat getEye()
{
    cv::Mat rot = cv::Mat::zeros(3, 3, CV_32FC1);
    rot.at<float>(0,0) = 1.0f; 
    rot.at<float>(1,0) = 0.0f; 
    rot.at<float>(2,0) = 0.0f;
    rot.at<float>(0,1) = 0.0f; 
    rot.at<float>(1,1) = 1.0f; 
    rot.at<float>(2,1) = 0.0f;
    rot.at<float>(0,2) = 0.0f; 
    rot.at<float>(1,2) = 0.0f; 
    rot.at<float>(2,2) = 1.0f;
    return rot;
}

std::vector<cv::Mat> getRotationMatrices(const std::vector<float> &rot_params) 
{
    // rotation matrix along z axis 
    std::vector<cv::Mat> rotation_matrices;
    std::vector<float> rot_params_z;  // yaw
    rot_params_z.push_back(-14.0f);  // center front
    rot_params_z.push_back(-119.0f);  // center rear
    rot_params_z.push_back(-43.65f);  // driver front (left front)
    rot_params_z.push_back(-125.0f); // driver side (left side)
    rot_params_z.push_back(38.5f);   // passenger front (right front)
    rot_params_z.push_back(40.0f);  // passenger side (right side)

    std::vector<float> rot_params_x; // pitch
    rot_params_x.push_back(-1.8f);  // center front
    rot_params_x.push_back(0.6f);  // center rear
    rot_params_x.push_back(-0.5f);  // driver front (left front)
    rot_params_x.push_back(-1.2f); // driver side (left side)
    rot_params_x.push_back(-0.2f);   // passenger front (right front)
    rot_params_x.push_back(0.0f);  // passenger side (right side)

    std::vector<float> rot_params_y; // roll
    rot_params_y.push_back(1.0f);  // center front
    rot_params_y.push_back(0.5f);  // center rear
    rot_params_y.push_back(0.0f);  // driver front (left front)
    rot_params_y.push_back(1.3f); // driver side (left side)
    rot_params_y.push_back(0.5f);   // passenger front (right front)
    rot_params_y.push_back(-0.2f);  // passenger side (right side)
    for (int i = 0; i < rot_params.size(); i++) 
    {
        // Multiply the rotation matrix if rotated along more than one axis
        // For example
        // cv::Mat rot = getRotationMatrixFromTheta_Z(rot_params[i]) * getRotationMatrixFromTheta_Z(rot_params[i]);
        // rotation_matrices.push_back(getRotationMatrixFromTheta_Z(rot_params[i])); 
        //rotation_matrices.push_back(getEye()); 
		cv::Mat rot = getRotationMatrixFromTheta_Z(rot_params_z[i]) * getRotationMatrixFromTheta_X(rot_params_x[i]) * getRotationMatrixFromTheta_Y(rot_params_y[i]);
        //rotation_matrices.push_back(getRotationMatrixFromTheta_Z(rot_params[i]));
        rotation_matrices.push_back(rot);
    }
    return rotation_matrices;
}

std::vector<cv::Mat> getTranslationMatrices() {
    std::vector<std::vector<float>> trans_params;
    trans_params.push_back({3.98f, 0.00f, -0.32f});  // center front
    trans_params.push_back({-1.19f, 0.00f, -0.17f});  // center rear
    trans_params.push_back({3.87f, -0.60f, -0.02f});  // driver front (left front)
    trans_params.push_back({2.70f, -0.90f, -0.55f}); // driver side (left side)
    trans_params.push_back({3.85f, 0.61f, -0.02f});   // passenger front (right front)
    trans_params.push_back({2.70f, 0.90f, -0.55f});  // passenger side (right side)
    // trans_params.push_back({3.98f, 0.00f, -0.22f});  // center front
    // trans_params.push_back({-1.19f, 0.00f, 0.14f});  // center rear
    // trans_params.push_back({3.91f, -0.60f, 0.00f});  // driver front (left front)
    // trans_params.push_back({2.70f, -0.90f, -0.55f}); // driver side (left side)
    // trans_params.push_back({3.83f, 0.61f, 0.00f});   // passenger front (right front)
    // trans_params.push_back({2.70f, 0.90f, -0.50f});  // passenger side (right side

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

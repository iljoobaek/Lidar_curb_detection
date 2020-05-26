#include <iostream>
#include <vector>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <experimental/filesystem>
#include <memory>

// Include VelodyneCapture Header
#include "VelodyneCapture.h"
namespace fs = std::experimental::filesystem;

namespace LidarViewer {
std::vector<std::string> get_file_names() {
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
    return files;
}

void update_viewer(std::vector<std::vector<cv::Vec3f>> &buffers, std::vector<std::vector<bool>> &results, cv::viz::Viz3d &viewer) {
    // if (buffers[0].empty()) {return;}
    cv::viz::WCloudCollection collection;
    std::vector<cv::Vec3f> curbsBuffer;
    for (int i = 0; i < buffers.size(); i++) {
        int idx = 0;
        for (int j = 0; j < buffers[i].size(); j++) {
            if (results[i][j]) {
                curbsBuffer.push_back(buffers[i][j]);
            } else {
                buffers[i][idx++] = buffers[i][j];
            }
        }
        buffers[i].resize(idx);
    }
    buffers.push_back(curbsBuffer); 
    
    for (int i = 0; i < buffers.size(); i++) {
        cv::Mat cloudMat = cv::Mat(static_cast<int>(buffers[i].size()), 1, CV_32FC3, &buffers[i][0]);
        if (i == buffers.size()-1) {
            collection.addCloud(cloudMat, cv::viz::Color::red());
        }
        else {
            collection.addCloud(cloudMat, cv::viz::Color::white());
        }
    }
    viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
    viewer.showWidget("Cloud", collection);
    viewer.setRenderingProperty("Cloud", cv::viz::POINT_SIZE, 2.0);  // Set point size of the point cloud
    viewer.setBackgroundColor(cv::viz::Color::mlab()); 
    viewer.spinOnce();
}

void update_viewer(std::vector<std::vector<cv::Vec3f>> &buffers, std::vector<std::vector<int>> &results, cv::viz::Viz3d &viewer) {
    // if (buffers[0].empty()) {return;}
    cv::viz::WCloudCollection collection;

    std::vector<cv::Vec3f> curbsBuffer_1;
    std::vector<cv::Vec3f> curbsBuffer_2;
    for (int i = 0; i < buffers.size(); i++) {
        int idx = 0;
        for (int j = 0; j < buffers[i].size(); j++) {
            if (results[i][j] == 1) {
                curbsBuffer_1.push_back(buffers[i][j]);
            } 
            else if (results[i][j] == -1) {
                curbsBuffer_2.push_back(buffers[i][j]);
            } 
            else {
                buffers[i][idx++] = buffers[i][j];
            }
        }
        buffers[i].resize(idx);
    }
    buffers.push_back(curbsBuffer_1); 
    buffers.push_back(curbsBuffer_2); 
    
    for (int i = 0; i < buffers.size(); i++) {
        cv::Mat cloudMat = cv::Mat(static_cast<int>(buffers[i].size()), 1, CV_32FC3, &buffers[i][0]);
        if (i == buffers.size()-1) {
            collection.addCloud(cloudMat, cv::viz::Color::red());
        }
        else if (i == buffers.size()-2) {
            collection.addCloud(cloudMat, cv::viz::Color::green());
        }
        else {
            collection.addCloud(cloudMat, cv::viz::Color::white());
        }
    }
    viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
    viewer.showWidget("Cloud", collection);
    viewer.setRenderingProperty("Cloud", cv::viz::POINT_SIZE, 2.0);  // Set point size of the point cloud
    viewer.setBackgroundColor(cv::viz::Color::mlab()); 
    viewer.spinOnce();
}

void update_viewer_one_buffer(std::vector<cv::Vec3f> &buffer, cv::viz::Viz3d &viewer) {
    cv::Mat cloudMat = cv::Mat(static_cast<int>(buffer.size()), 1, CV_32FC3, &buffer[0]);
    cv::viz::WCloud cloud( cloudMat );
    viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
    viewer.showWidget("Cloud", cloud);
    viewer.spinOnce();
}

std::vector<float> get_rot_params() {
    std::vector<float> rot_params;
    rot_params.push_back(-14.0f);  // center front
    rot_params.push_back(-119.0f);  // center rear
    rot_params.push_back(-44.0f);  // driver front (left front)
    rot_params.push_back(-125.0f); // driver side (left side)
    rot_params.push_back(38.5f);   // passenger front (right front)
    rot_params.push_back(40.0f);  // passenger side (right side)
    return rot_params;
}

std::vector<cv::Mat> get_rotation_matrices(const std::vector<float> &rot_params) {
    // rotation matrix along z axis 
    std::vector<cv::Mat> rotation_matrices;
    for (int i = 0; i < rot_params.size(); i++) {
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

std::vector<cv::Mat> get_translation_matrices() {
    std::vector<std::vector<float>> trans_params;
    trans_params.push_back({3.98f, 0.00f, -0.22f});  // center front
    trans_params.push_back({-1.19f, 0.00f, 0.14f});  // center rear
    trans_params.push_back({3.91f, -0.60f, 0.00f});  // driver front (left front)
    trans_params.push_back({2.70f, -0.90f, -0.55f}); // driver side (left side)
    trans_params.push_back({3.83f, 0.61f, 0.00f});   // passenger front (right front)
    trans_params.push_back({2.70f, 0.90f, -0.50f});  // passenger side (right side)

    std::vector<cv::Mat> translation_matrices;
    for (int i = 0; i < trans_params.size(); i++) {
        cv::Mat trans = cv::Mat::zeros(3, 1, CV_32FC1);
        trans.at<float>(0,0) = trans_params[i][0]; 
        trans.at<float>(1,0) = trans_params[i][1]; 
        trans.at<float>(2,0) = trans_params[i][2];
        translation_matrices.push_back(trans); 
    }
    return translation_matrices;
}

void laser_to_cartesian(std::vector<velodyne::Laser> &lasers, std::vector<std::vector<float>> &pointcloud, float theta, cv::Mat &rot, cv::Mat &trans) {
    pointcloud.clear();
    pointcloud.resize(lasers.size());
    int idx = 0;
    for (int i = 0; i < lasers.size(); i++) {
        const double distance = static_cast<double>( lasers[i].distance );
        const double azimuth  = lasers[i].azimuth  * CV_PI / 180.0;
        const double vertical = lasers[i].vertical * CV_PI / 180.0;
        float x = static_cast<float>( ( distance * std::cos( vertical ) ) * std::sin( azimuth ) );
        float y = static_cast<float>( ( distance * std::cos( vertical ) ) * std::cos( azimuth ) );
        float z = static_cast<float>( ( distance * std::sin( vertical ) ) );
        
        if( x == 0.0f && y == 0.0f && z == 0.0f ) continue;

        x /= 100.0, y /= 100.0, z /= 100.0;
        float intensity = static_cast<float>(lasers[i].intensity);
        float ring = static_cast<float>(lasers[i].id);
        float dist = std::sqrt(x * x + y * y + z * z);
        if (dist < 0.9f) continue;
        float azimuth_rot = static_cast<float>(lasers[i].azimuth) + theta;
        if (azimuth_rot >= 360.0f) {
            azimuth_rot -= 360.0f;
        }
        else if (azimuth_rot < 0.0f) {
            azimuth_rot += 360.0f;
        }
        pointcloud[idx] = {x, y, z, intensity, ring, dist, azimuth_rot}; // Write to pointcloud
        idx++;
    }
    pointcloud.resize(idx);
}

void push_result_to_buffer(std::vector<cv::Vec3f> &buffer, const std::vector<std::vector<float>> &pointcloud, cv::Mat &rot, cv::Mat &trans) {
    buffer.resize(pointcloud.size());
    for (int i = 0; i < pointcloud.size(); i++) {
        buffer[i] = cv::Vec3f( pointcloud[i][0], pointcloud[i][1], pointcloud[i][2] ); 
        // cv::Mat p(buffer[i]);
        // float temp_x = p.at<float>(0,0), temp_y = p.at<float>(1,0); 
        // buffer[i][0] = temp_y + trans.at<float>(0,0);
        // buffer[i][1] = temp_x + trans.at<float>(1,0);
        // buffer[i][2] = -buffer[i][2] + trans.at<float>(2,0);
    }
}
}
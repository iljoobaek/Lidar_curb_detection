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

namespace LidarViewer 
{
void cvViz3dCallbackSetting(cv::viz::Viz3d &viewer, bool &pause)
{
    // Register Keyboard Callback
    viewer.registerKeyboardCallback(
        []( const cv::viz::KeyboardEvent& event, void* cookie )
        {
        // Close Viewer
        if( event.code == 'q' && event.action == cv::viz::KeyboardEvent::Action::KEY_DOWN )
        {
            static_cast<cv::viz::Viz3d*>( cookie )->close();
        }
        }
        , &viewer);
    viewer.registerKeyboardCallback(
        []( const cv::viz::KeyboardEvent& event, void* pause )
        {
        // Switch state of pause / resume when pressing p
        if( event.code == 'p' && event.action == cv::viz::KeyboardEvent::Action::KEY_DOWN )
        {
            bool* p = static_cast<bool*>( pause );
            *p = !(*p);
        }
        }
        , &pause);
}

void updateViewerFromBuffers(std::vector<std::vector<cv::Vec3f>> &buffers, std::vector<std::vector<bool>> &results, cv::viz::Viz3d &viewer) 
{
    // if (buffers[0].empty()) {return;}
    cv::viz::WCloudCollection collection;
    std::vector<cv::Vec3f> curbsBuffer;
    for (int i = 0; i < buffers.size(); i++) 
    {
        int idx = 0;
        for (int j = 0; j < buffers[i].size(); j++) 
        {
            if (results[i][j]) 
            {
                curbsBuffer.push_back(buffers[i][j]);
            } else 
            {
                buffers[i][idx++] = buffers[i][j];
            }
        }
        buffers[i].resize(idx);
    }
    buffers.push_back(curbsBuffer); 
    
    for (int i = 0; i < buffers.size(); i++) 
    {
        cv::Mat cloudMat = cv::Mat(static_cast<int>(buffers[i].size()), 1, CV_32FC3, &buffers[i][0]);
        if (i == buffers.size()-1) 
        {
            collection.addCloud(cloudMat, cv::viz::Color::red());
        }
        else 
        {
            collection.addCloud(cloudMat, cv::viz::Color::white());
        }
    }
    viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
    viewer.showWidget("Cloud", collection);
    viewer.setRenderingProperty("Cloud", cv::viz::POINT_SIZE, 2.0);  // Set point size of the point cloud
    viewer.setBackgroundColor(cv::viz::Color::mlab()); 
    viewer.spinOnce();
}

void updateViewerFromBuffers(std::vector<std::vector<cv::Vec3f>> &buffers, std::vector<std::vector<int>> &results, cv::viz::Viz3d &viewer) {
    // if (buffers[0].empty()) {return;}
    cv::viz::WCloudCollection collection;

    std::vector<cv::Vec3f> curbsBuffer_1;
    std::vector<cv::Vec3f> curbsBuffer_2;
    for (int i = 0; i < buffers.size(); i++) 
    {
        int idx = 0;
        for (int j = 0; j < buffers[i].size(); j++) 
        {
            if (results[i][j] == 1) 
            {
                curbsBuffer_1.push_back(buffers[i][j]);
            } 
            else if (results[i][j] == -1) 
            {
                curbsBuffer_2.push_back(buffers[i][j]);
            } 
            else 
            {
                buffers[i][idx++] = buffers[i][j];
            }
        }
        buffers[i].resize(idx);
    }
    buffers.push_back(curbsBuffer_1); 
    buffers.push_back(curbsBuffer_2); 
    
    for (int i = 0; i < buffers.size(); i++) 
    {
        cv::Mat cloudMat = cv::Mat(static_cast<int>(buffers[i].size()), 1, CV_32FC3, &buffers[i][0]);
        if (i == buffers.size()-1) 
        {
            collection.addCloud(cloudMat, cv::viz::Color::red());
        }
        else if (i == buffers.size()-2) 
        {
            collection.addCloud(cloudMat, cv::viz::Color::green());
        }
        else 
        {
            collection.addCloud(cloudMat, cv::viz::Color::white());
        }
    }
    viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
    viewer.showWidget("Cloud", collection);
    viewer.setRenderingProperty("Cloud", cv::viz::POINT_SIZE, 2.0);  // Set point size of the point cloud
    viewer.setBackgroundColor(cv::viz::Color::mlab()); 
    viewer.spinOnce();
}

void updateViewerFromSingleBuffer(std::vector<cv::Vec3f> &buffer, cv::viz::Viz3d &viewer) 
{
    cv::Mat cloudMat = cv::Mat(static_cast<int>(buffer.size()), 1, CV_32FC3, &buffer[0]);
    cv::viz::WCloud cloud( cloudMat );
    viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
    viewer.showWidget("Cloud", cloud);
    viewer.spinOnce();
}

void laser_to_cartesian(std::vector<velodyne::Laser> &lasers, std::vector<std::vector<float>> &pointcloud, float theta, cv::Mat &rot, cv::Mat &trans) 
{
    pointcloud.clear();
    pointcloud.resize(lasers.size());
    int idx = 0;
    for (int i = 0; i < lasers.size(); i++) 
    {
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
        if (azimuth_rot >= 360.0f) 
        {
            azimuth_rot -= 360.0f;
        }
        else if (azimuth_rot < 0.0f) 
        {
            azimuth_rot += 360.0f;
        }
        pointcloud[idx] = {x, y, z, intensity, ring, dist, azimuth_rot}; // Write to pointcloud
        idx++;
    }
    pointcloud.resize(idx);
}

void push_result_to_buffer(std::vector<cv::Vec3f> &buffer, const std::vector<std::vector<float>> &pointcloud, cv::Mat &rot, cv::Mat &trans) 
{
    buffer.resize(pointcloud.size());
    for (int i = 0; i < pointcloud.size(); i++) 
    {
        buffer[i] = cv::Vec3f( pointcloud[i][0], pointcloud[i][1], pointcloud[i][2] ); 
        // cv::Mat p(buffer[i]);
        // float temp_x = p.at<float>(0,0), temp_y = p.at<float>(1,0); 
        // buffer[i][0] = temp_y + trans.at<float>(0,0);
        // buffer[i][1] = temp_x + trans.at<float>(1,0);
        // buffer[i][2] = -buffer[i][2] + trans.at<float>(2,0);
    }
}
}
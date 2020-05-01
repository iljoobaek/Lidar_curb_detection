#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

// Include VelodyneCapture Header
#include "VelodyneCapture.h"

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

// Test the results coming with type int
void updateViewerFromBuffers(std::vector<std::vector<cv::Vec3f>> &buffers, cv::viz::Viz3d &viewer, std::vector<std::vector<float>> &vscanRes) 
{
    if (buffers[0].empty()) {return;}
    cv::viz::WCloudCollection collection;

	int cnt = 0; 
    for (auto &res : vscanRes)
    {
        cv::viz::WLine line(cv::Point3f(res[0], res[1], res[2]), cv::Point3f(res[0], res[1], res[3]), cv::viz::Color::green());
        line.setRenderingProperty(cv::viz::LINE_WIDTH, 5.0); 
        viewer.showWidget("Line Widget"+std::to_string(cnt++), line);
    }

    cv::Mat cloudMat = cv::Mat(static_cast<int>(buffers[0].size()), 1, CV_32FC3, &buffers[0][0]);
    collection.addCloud(cloudMat, cv::viz::Color::white());

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

void pushToBuffer(std::vector<cv::Vec3f> &buffer, const std::vector<std::vector<float>> &pointcloud) 
{
    buffer.resize(pointcloud.size());
    for (int i = 0; i < pointcloud.size(); i++) 
    {
        buffer[i] = cv::Vec3f( pointcloud[i][0], pointcloud[i][1], pointcloud[i][2] ); 
    }
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

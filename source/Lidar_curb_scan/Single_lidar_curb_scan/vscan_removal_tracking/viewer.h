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
    
// Manoj changes:
// Visualize vscan result, ground truth lines and detected lines
void updateViewerFromBuffers(std::vector<std::vector<cv::Vec3f>> &buffers, std::vector<std::vector<bool>> &results, cv::viz::Viz3d &viewer, std::vector<std::vector<float>> &vscanRes, std::vector<cv::viz::WPolyLine> &polyLine, std::vector<cv::viz::WPolyLine> &gtLines, std::vector<cv::Vec3f> transformedL, std::vector<cv::Vec3f> transformedR, std::vector<cv::Vec3f> ultrasonicPoints) 
{
    // if (buffers[0].empty()) {return;}
    viewer.removeAllWidgets();
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
    //buffers.push_back(curbsBuffer); 
    std::vector<cv::viz::WLine> lines;
    int cnt = 0; 
    for (auto &res : vscanRes)
    {
        cv::viz::WLine line(cv::Point3f(res[0], res[1], res[2]), cv::Point3f(res[0], res[1], res[3]), cv::viz::Color::green());
        line.setRenderingProperty(cv::viz::LINE_WIDTH, 5.0); 
        viewer.showWidget("Line Widget"+std::to_string(cnt++), line);
    }


    for (int i = 0; i < buffers.size(); i++) 
    {
        cv::Mat cloudMat = cv::Mat(static_cast<int>(buffers[i].size()), 1, CV_32FC3, &buffers[i][0]);
        if (i == buffers.size()-1) 
        {
            collection.addCloud(cloudMat, cv::viz::Color::red());
            
        }
       /* else if (i == buffers.size()-2) 
        {
            collection.addCloud(cloudMat, cv::viz::Color::red());
            
        } */

        else 
        {
            collection.addCloud(cloudMat, cv::viz::Color::white());
        }
    }

	for (int i = 0; i < ultrasonicPoints.size(); i++) {
		cv::Mat ultrasonicMat = cv::Mat(static_cast<int>(ultrasonicPoints.size()), 1, CV_32FC3, &ultrasonicPoints[0]);
			collection.addCloud(ultrasonicMat, cv::viz::Color::brown()); //Ultrasonic - pink
	} 
    for (int i = 0; i < transformedL.size(); i++){
        cv::Mat transformedLMat = cv::Mat(static_cast<int>(transformedL.size()), 1, CV_32FC3, &transformedL[0]);
        if (i == buffers.size()-1) 
        {
            collection.addCloud(transformedLMat, cv::viz::Color::cyan()); //L - purple
            
        }
    }

    for (int i = 0; i < transformedR.size(); i++){
        cv::Mat transformedRMat = cv::Mat(static_cast<int>(transformedR.size()), 1, CV_32FC3, &transformedR[0]);
        if (i == buffers.size()-1) 
        {
            collection.addCloud(transformedRMat, cv::viz::Color::orange()); //R - Orange
            
        }
    }
    
    polyLine[0].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0); 
    polyLine[1].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0); 
    polyLine[2].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0); // left
    polyLine[3].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0); // right
    gtLines[0].setRenderingProperty(cv::viz::LINE_WIDTH, 2.0); 
    gtLines[1].setRenderingProperty(cv::viz::LINE_WIDTH, 2.0); 
    viewer.showWidget("Poly Left", polyLine[0]);
    viewer.showWidget("Poly Right", polyLine[1]);
    viewer.showWidget("Poly newleft", polyLine[2]);
    viewer.showWidget("Poly newright", polyLine[3]);
    viewer.showWidget("Poly Left gt", gtLines[0]);
    viewer.showWidget("Poly Right gt", gtLines[1]);
    viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
    viewer.showWidget("Cloud", collection);
    viewer.setRenderingProperty("Cloud", cv::viz::POINT_SIZE, 3.0);  // Set point size of the point cloud
    viewer.setBackgroundColor(cv::viz::Color::mlab()); 
    viewer.spinOnce();
}
    
    
// Visualize vscan result, ground truth lines and detected lines
void updateViewerFromBuffersOnlyLidar(std::vector<std::vector<cv::Vec3f>> &buffers, cv::viz::Viz3d &viewer) 
{
    // if (buffers[0].empty()) {return;}
    viewer.removeAllWidgets();
    cv::viz::WCloudCollection collection;
    for (int i = 0; i < buffers.size(); i++) 
    {
        int idx = 0;
        for (int j = 0; j < buffers[i].size(); j++) 
        {
                buffers[i][idx++] = buffers[i][j];
        }
        buffers[i].resize(idx);
    }

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

// Test the results coming with type int
void updateViewerFromBuffers(std::vector<std::vector<cv::Vec3f>> &buffers, std::vector<std::vector<int>> &results, cv::viz::Viz3d &viewer, std::vector<std::vector<float>> &vscanRes) 
{
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
    
    for (auto &res : vscanRes)
    {
        std::cout << res[0] << " " << res[1] << " " << res[2] << " " << res[3] << std::endl;
        cv::viz::WLine line(cv::Point3f(res[0], res[1], res[2]), cv::Point3f(res[0], res[1], res[3]), cv::viz::Color::green());
        line.setRenderingProperty(cv::viz::LINE_WIDTH, 5.0); 
        viewer.showWidget("Line Widget", line);
    }




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

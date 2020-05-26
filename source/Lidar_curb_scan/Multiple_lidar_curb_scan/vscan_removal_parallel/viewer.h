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


// void updateViewerFromBuffersManager(std::vector<std::vector<cv::Vec3f>> &buffers, cv::viz::Viz3d &viewer,\
//     std::vector<cv::viz::WPolyLine> &polyLine, std::vector<cv::Vec3f> transformedL, std::vector<cv::Vec3f> transformedR){
        
//     viewer.removeAllWidgets();
//     cv::viz::WCloudCollection collection;
//     int cnt = 0; 
//     std::cout << " Starting viewer \n";
//     // for (auto &res : vscanRes) // VScan
//     // {
//     //     std::cout << "fin "<<res[0]<<"\n";
//     //     std::cout << "fin "<<res[3]<<"\n";
//     //     cv::viz::WLine line(cv::Point3f(res[0], res[1], res[2]), cv::Point3f(res[0], res[1], res[3]), cv::viz::Color::green());
//     //     line.setRenderingProperty(cv::viz::LINE_WIDTH, 5.0); 
//     //     viewer.showWidget("Line Widget"+std::to_string(cnt++), line);
//     // }

//     // for(int i=0; i<vscanRes.size(); i++){
//     //     std::cout << "fin "<<vscanRes[i][0]<<"\n";
//     //     cv::viz::WLine line(cv::Point3f(&vscanRes[i][0], &vscanRes[i][1], &vscanRes[i][2]), cv::Point3f(&vscanRes[i][0], &vscanRes[i][1], &vscanRes[i][3], cv::viz::Color::green());
//     //     std::cout << "able to push\n";
//     // }

//     std::cout << "Updated Vscan \n";
//     for (int i = 0; i < buffers.size(); i++) // Pointcloud 
//     {
//         std::cout << " Pushing pcd\n";
//         cv::Mat cloudMat = cv::Mat(static_cast<int>(buffers[i].size()), 1, CV_32FC3, &buffers[i][0]);
        
//         if (i == 0){
//             std::cout << "Adding red \n";
//             collection.addCloud(cloudMat, cv::viz::Color::red()); // Front 
//         }else{
//             if(i == 1){
//                 std::cout << "Adding White \n";
//                 collection.addCloud(cloudMat, cv::viz::Color::white());
//             }
//             else{
//                 collection.addCloud(cloudMat, cv::viz::Color::blue());
//             }
//         }            
//     }
//     std::cout << "Uodated buffers \n";
//     for (int i = 0; i < transformedL.size(); i++) // Left curb
//     {
//         std::cout << "trnformed l \n";
//         cv::Mat transformedLMat = cv::Mat(static_cast<int>(transformedL.size()), 1, CV_32FC3, &transformedL[0]);
//         if (i == 0){
//             std::cout << "trnformed l Adding\n";
//             collection.addCloud(transformedLMat, cv::viz::Color::cyan()); //Left - Cyan
//         } 
            
//     }
//     for (int i = 0; i < transformedR.size(); i++) // Right curb
//     {
//         std::cout << "trnformed R \n";
//         cv::Mat transformedRMat = cv::Mat(static_cast<int>(transformedR.size()), 1, CV_32FC3, &transformedR[0]);
//         if (i == 0){
//             std::cout << "trnformed R Adding\n";
//             collection.addCloud(transformedRMat, cv::viz::Color::orange()); //Right - Orange
//         } 
            
//     }
//     polyLine[0].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0); 
//     polyLine[1].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0); 
//     viewer.showWidget("Poly Left", polyLine[0]);
//     viewer.showWidget("Poly Right", polyLine[1]);
//     viewer.showWidget("Cloud", collection);
//     viewer.setRenderingProperty("Cloud", cv::viz::POINT_SIZE, 3.0);  // Set point size of the point cloud
//     viewer.setBackgroundColor(cv::viz::Color::mlab()); // Background color
//     viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
//     viewer.spinOnce();
// }
    

void laser_to_cartesian(std::vector<velodyne::Laser> &lasers, std::vector<std::vector<float>> &pointcloud, float theta, cv::Mat &rot, cv::Mat &trans) {
    pointcloud.clear();
    pointcloud.resize(lasers.size());
    int idx = 0;
    for (int i = 0; i < lasers.size(); i++) {
        //double azimuth_rot = lasers[i].azimuth + theta;
        double azimuth_rot = lasers[i].azimuth;
        if (azimuth_rot >= 360.0) {
            azimuth_rot -= 360.0;
        }
        else if (azimuth_rot < 0.0) {
            azimuth_rot += 360.0;
        }
        const double distance = static_cast<double>( lasers[i].distance );
        const double azimuth  = azimuth_rot  * CV_PI / 180.0;
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
        
        pointcloud[idx] = {x, y, z, intensity, ring, dist, static_cast<float>(azimuth_rot)}; // Write to pointcloud
        idx++;
    }
    pointcloud.resize(idx);
}

void updateViewerFromBuffersManager(std::vector<std::vector<cv::Vec3f>> &buffers, cv::viz::Viz3d &viewer, std::vector<std::vector<float>> &vscanRes, \
    std::vector<cv::viz::WPolyLine> &polyLine, std::vector<cv::viz::WPolyLine> &polyLine2, std::vector<cv::Vec3f> transformedL, std::vector<cv::Vec3f> transformedR, \
    std::vector<cv::Vec3f> transformedL2, std::vector<cv::Vec3f> transformedR2){
    viewer.removeAllWidgets();

    cv::viz::WCloudCollection collection;
    int cnt = 0; 
    std::cout << " Starting viewer \n";
        for (auto res : vscanRes) // VScan
        {
                cv::viz::WLine line(cv::Point3f(res[0], res[1], res[2]), cv::Point3f(res[0], res[1], res[3]), cv::viz::Color::green());
                line.setRenderingProperty(cv::viz::LINE_WIDTH, 5.0); 
                viewer.showWidget("Line Widget"+std::to_string(cnt++), line);
        }
    
    std::cout << "Updated Vscan \n";
    
    for (int i = 0; i < buffers.size(); i++) 
    {
        int idx = 0;
        for (int j = 0; j < buffers[i].size(); j++) 
        {
                buffers[i][idx++] = buffers[i][j];
        }
        buffers[i].resize(idx);
    }
    std::cout << "Updated pcd \n"; 

    for (int i = 0; i < buffers.size(); i++) // Pointcloud 
    {
        std::cout << " Pushing pcd\n";
        cv::Mat cloudMat = cv::Mat(static_cast<int>(buffers[i].size()), 1, CV_32FC3, &buffers[i][0]);
        
        if (i == 1){
            std::cout << "Adding red \n";
            collection.addCloud(cloudMat, cv::viz::Color::white()); // Front 
        }else{
            if(i == 0){
                std::cout << "Adding White \n";
                collection.addCloud(cloudMat, cv::viz::Color::white());
            }
            else{
                collection.addCloud(cloudMat, cv::viz::Color::cyan());
            }
        }            
    }
    std::cout << "Uodated buffers \n";
    std::cout << "trnformed l "<<transformedL.size()<<"\n";
    cv::Mat transformedLMat = cv::Mat(static_cast<int>(transformedL.size()), 1, CV_32FC3, &transformedL[0]);
    std::cout << "trnformed l Adding\n";
    collection.addCloud(transformedLMat, cv::viz::Color::cyan()); //Left - Cyan


    std::cout << "trnformed R \n";
    std::cout << "trnformed l "<<transformedR.size()<<"\n";
    cv::Mat transformedRMat = cv::Mat(static_cast<int>(transformedR.size()), 1, CV_32FC3, &transformedR[0]);
    std::cout << "trnformed R Adding\n";
    collection.addCloud(transformedRMat, cv::viz::Color::orange()); //Right - Orange

    std::cout << "Uodated buffers \n";
    std::cout << "trnformed l "<<transformedL2.size()<<"\n";
    cv::Mat transformedL2Mat = cv::Mat(static_cast<int>(transformedL2.size()), 1, CV_32FC3, &transformedL2[0]);
    std::cout << "trnformed l Adding\n";
    collection.addCloud(transformedL2Mat, cv::viz::Color::cyan()); //Left - Cyan


    std::cout << "trnformed R \n";
    std::cout << "trnformed l "<<transformedR2.size()<<"\n";
    cv::Mat transformedR2Mat = cv::Mat(static_cast<int>(transformedR2.size()), 1, CV_32FC3, &transformedR2[0]);
    std::cout << "trnformed R Adding\n";
    collection.addCloud(transformedR2Mat, cv::viz::Color::orange()); //Right - Orange


    polyLine[0].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0); 
    polyLine[1].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0); 
    polyLine2[0].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0); 
    polyLine2[1].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0); 
    viewer.showWidget("Poly Left", polyLine[0]);
    viewer.showWidget("Poly Right", polyLine[1]);
    viewer.showWidget("Poly Left2", polyLine2[0]);
    viewer.showWidget("Poly Right2", polyLine2[1]);
    viewer.showWidget("Cloud", collection);
    viewer.setRenderingProperty("Cloud", cv::viz::POINT_SIZE, 3.0);  // Set point size of the point cloud
    viewer.setBackgroundColor(cv::viz::Color::mlab()); // Background color
    viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
    viewer.spinOnce();
}
    
// Manoj changes:
// Visualize vscan result, ground truth lines and detected lines
void updateViewerFromBuffers(std::vector<std::vector<cv::Vec3f>> &buffers, std::vector<std::vector<bool>> &results, cv::viz::Viz3d &viewer, std::vector<std::vector<float>> &vscanRes, std::vector<cv::viz::WPolyLine> &polyLine, std::vector<cv::viz::WPolyLine> &gtLines, 
std::vector<cv::Vec3f> transformedL, std::vector<cv::Vec3f> transformedR, std::vector<cv::Vec3f> ultrasonicPoints) 
{
    // if (buffers[0].empty()) {return;}
    viewer.removeAllWidgets();
    cv::viz::WCloudCollection collection;
    std::vector<cv::Vec3f> curbsBuffer;
    
    std::cout<<"Starting buffer loop \n"; 
    
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
    
    std::cout<<"got buffers \n"; 
    
    for (auto &res : vscanRes)
    {
        cv::viz::WLine line(cv::Point3f(res[0], res[1], res[2]), cv::Point3f(res[0], res[1], res[3]), cv::viz::Color::green());
        line.setRenderingProperty(cv::viz::LINE_WIDTH, 5.0); 
        viewer.showWidget("Line Widget"+std::to_string(cnt++), line);
    }

    std::cout<<"got showWidget line \n"; 

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
    
    std::cout<<"got addCloud \n"; 

    for (int i = 0; i < ultrasonicPoints.size(); i++) {
        cv::Mat ultrasonicMat = cv::Mat(static_cast<int>(ultrasonicPoints.size()), 1, CV_32FC3, &ultrasonicPoints[0]);
            collection.addCloud(ultrasonicMat, cv::viz::Color::brown()); //Ultrasonic - pink
    }
    
    std::cout<<"got ultrasonicMat \n";
    
    for (int i = 0; i < transformedL.size(); i++){
        cv::Mat transformedLMat = cv::Mat(static_cast<int>(transformedL.size()), 1, CV_32FC3, &transformedL[0]);
        if (i == buffers.size()-1) 
        {
            collection.addCloud(transformedLMat, cv::viz::Color::cyan()); //L - purple
            
        }
    }
    
    std::cout<<"got transformedLMat \n";

    for (int i = 0; i < transformedR.size(); i++){
        cv::Mat transformedRMat = cv::Mat(static_cast<int>(transformedR.size()), 1, CV_32FC3, &transformedR[0]);
        if (i == buffers.size()-1) 
        {
            collection.addCloud(transformedRMat, cv::viz::Color::orange()); //R - Orange
            
        }
    }
    
    std::cout<<"got transformedRMat \n";
    
    polyLine[0].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0); 
    polyLine[1].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0); 
//     polyLine[2].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0); // left
//     polyLine[3].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0); // right
//     gtLines[0].setRenderingProperty(cv::viz::LINE_WIDTH, 2.0); 
//     gtLines[1].setRenderingProperty(cv::viz::LINE_WIDTH, 2.0); 
    
    std::cout<<"got setRenderingProperty \n";
    
    viewer.showWidget("Poly Left", polyLine[0]);
    viewer.showWidget("Poly Right", polyLine[1]);
//     viewer.showWidget("Poly newleft", polyLine[2]);
//     viewer.showWidget("Poly newright", polyLine[3]);
//     viewer.showWidget("Poly Left gt", gtLines[0]);
//     viewer.showWidget("Poly Right gt", gtLines[1]);
    viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
    viewer.showWidget("Cloud", collection);
    
    std::cout<<"got showWidget \n";
    
    viewer.setRenderingProperty("Cloud", cv::viz::POINT_SIZE, 3.0);  // Set point size of the point cloud
    viewer.setBackgroundColor(cv::viz::Color::mlab()); 
    
    std::cout<<"got setBackgroundColor \n";
    
    viewer.spinOnce();
    
    std::cout<<"got spinOnce \n";
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


// Visualize vscan result, ground truth lines and detected lines
// void updateViewerFromBuffersBoundary(std::vector<std::vector<cv::Vec3f>> &buffers, std::vector<std::vector<float>> &vscanRes, cv::viz::Viz3d &viewer) //std::vector<std::vector<cv::Vec3f>>&boundary_buffer,
void updateViewerFromBuffersBoundary(std::vector<std::vector<cv::Vec3f>> &buffers, std::vector<std::vector<bool>> &results, std::vector<std::vector<float>> &vscanRes, cv::viz::Viz3d &viewer) //std::vector<std::vector<cv::Vec3f>>boundary_buffer, 

{
    // if (buffers[0].empty()) {return;}
    viewer.removeAllWidgets();
    cv::viz::WCloudCollection collection;

    std::vector<cv::Vec3f> curbsBuffer;
    
    std::cout<<"Starting buffer loop \n"; 
    std::cout<<"Size of buffer "<<buffers[0].size()<<"\n"; 
    std::cout<<"Size of results "<<results[0].size()<<"\n"; 
    for (int i = 0; i < buffers.size(); i++) 
    {
        std::cout<<"i "<<i<<"\t"; 
        int idx = 0;
        for (int j = 0; j < buffers[i].size(); j++) 
        {
            std::cout<<"j "<<j<<"\t"; 
            if (results[i][j]) 
            {
                std::cout<<"result\t"; 
                curbsBuffer.push_back(buffers[i][j]);
            } else 
            {
                buffers[i][idx++] = buffers[i][j];
            }
            std::cout<<"Done\n"; 
        }
        std::cout<<"resizing\n"; 
        buffers[i].resize(idx);
    }
    std::cout<<"pushing \n"; 
    buffers.push_back(curbsBuffer); 



    // for (int i = 0; i < buffers.size(); i++) 
    // {
    //     int idx = 0;
    //     for (int j = 0; j < buffers[i].size(); j++) 
    //     {
    //             buffers[i][idx++] = buffers[i][j];
    //     }
    //     buffers[i].resize(idx);
    // }
    std::cout << "ciew pcd \n";
    for (int i = 0; i < buffers.size(); i++) 
    {
        cv::Mat cloudMat = cv::Mat(static_cast<int>(buffers[i].size()), 1, CV_32FC3, &buffers[i][0]);
        if (i == buffers.size()-1) 
        {
            collection.addCloud(cloudMat, cv::viz::Color::cyan());
            
        }
        else 
        {
            collection.addCloud(cloudMat, cv::viz::Color::white());
        }
    }    
    std::cout << "Vscan lines \n";
    int cnt = 0;
    for (auto res : vscanRes) // VScan
        {
                cv::viz::WLine line(cv::Point3f(res[0], res[1], res[2]), cv::Point3f(res[0], res[1], res[3]), cv::viz::Color::green());
                line.setRenderingProperty(cv::viz::LINE_WIDTH, 5.0); 
                viewer.showWidget("Line Widget"+std::to_string(cnt++), line);
        }

    // std::cout << "Vscan done, now boundary \n";
    // std::cout << "B1 size "<<boundary_buffer[0].size()<<"\n";
    // for (int i = 0; i < boundary_buffer[0].size(); i++){
    //     cv::Mat transformedLMat = cv::Mat(static_cast<int>(boundary_buffer[0].size()), 1, CV_32FC3, &boundary_buffer[0][0]);
    //     if (i == buffers.size()-1) 
    //     {
    //         collection.addCloud(transformedLMat, cv::viz::Color::red()); //L - purple
            
    //     }
    // }
    // std::cout << "B2 size "<<boundary_buffer[1].size()<<"\n";
    // std::cout<<"got transformedLMat \n";
    
    // for (int i = 0; i < boundary_buffer[1].size(); i++){
    //     cv::Mat transformedRMat = cv::Mat(static_cast<int>(boundary_buffer[1].size()), 1, CV_32FC3, &boundary_buffer[1][0]);
    //     if (i == buffers.size()-1) 
    //     {
    //         collection.addCloud(transformedRMat, cv::viz::Color::orange()); //R - Orange
            
    //     }
    // }
    
    std::cout<<"got transformedRMat \n";
    
    viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
    viewer.showWidget("Cloud", collection);
    viewer.setRenderingProperty("Cloud", cv::viz::POINT_SIZE, 2.0);  // Set point size of the point cloud
    viewer.setBackgroundColor(cv::viz::Color::mlab()); 
    viewer.spinOnce();
}





void updateViewerFromBuffersManagerAll(std::vector<std::vector<cv::Vec3f>> &buffers, std::vector<std::vector<cv::Vec3f>>&boundary_buffer,  std::vector<std::vector<std::vector<float>>> &vscanRes,  std::vector<std::vector<cv::viz::WPolyLine>> &polyLines, cv::viz::Viz3d &viewer) //std::vector<std::vector<cv::Vec3f>>boundary_buffer, 

{
    viewer.removeAllWidgets();
    cv::viz::WCloudCollection collection;

    std::vector<cv::Vec3f> curbsBuffer;
    
    std::cout<<"Starting buffer loop \n";

    for (int i = 0; i < buffers.size(); i++) 
    {
        int idx = 0;
        for (int j = 0; j < buffers[i].size(); j++) 
        {
                buffers[i][idx++] = buffers[i][j];
        }
        buffers[i].resize(idx);
    }
    std::cout << "ciew pcd \n";
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

    std::set <int, std::greater<int>> skip_vscan;
    for(int i=0; i<boundary_buffer.size(); i++){
        std::cout << "[VIEW] Boundary size: "<<boundary_buffer[i].size()<<"\n";
        if(boundary_buffer[i].size() < 5) 
        {
            skip_vscan.insert(i);
            std::cout << "[VIEW] Skipping "<<i<<"\n";
            continue;
        }
        cv::Mat transformedRMat = cv::Mat(static_cast<int>(boundary_buffer[i].size()), 1, CV_32FC3, &boundary_buffer[i][0]);
        std::cout << "[VIEW] Adding\n";
        collection.addCloud(transformedRMat, cv::viz::Color::orange()); //Right - Orange
    }


    int cnt = 0;
    for(int i=0; i<vscanRes.size(); i++){
        std::cout << "[VIEW] Vscan lines "<<vscanRes[i].size() <<"\n";
        if (skip_vscan.find(i) == skip_vscan.end()){
            for (auto res : vscanRes[i]) // VScan
            {
                std::cout << "[VIEW] Adding "<<i <<"\n";
                if(res.size() < 4) continue;
                std::cout << "[VIEW] Checking size "<<res.size()<<"\n";
                std::cout << "[VIEW] Adding "<<res[0]<<" "<<res[1]<<" "<<res[2]<<" "<<res[3]<<"\n";
                cv::viz::WLine line(cv::Point3f(res[0], res[1], res[2]), cv::Point3f(res[0], res[1], res[3]), cv::viz::Color::green());
                line.setRenderingProperty(cv::viz::LINE_WIDTH, 5.0); 
                viewer.showWidget("Line Widget"+std::to_string(cnt++), line);
            }
        }
    }
    
    std::cout << "[VIEW] Rendering  \n";
    std::string lineName;
    cnt = 0;
    for(auto polyLine: polyLines){
        polyLine[0].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0); 
        polyLine[1].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0);
        lineName = "Poly Left "+ std::to_string(cnt);
        viewer.showWidget(lineName, polyLine[0]);
        lineName = "Poly Right "+ std::to_string(cnt);
        viewer.showWidget(lineName, polyLine[1]); 
        cnt++;  
    }

    viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
    viewer.showWidget("Cloud", collection);
    
    viewer.setRenderingProperty("Cloud", cv::viz::POINT_SIZE, 2.0);  // Set point size of the point cloud
    viewer.setBackgroundColor(cv::viz::Color::mlab()); 
    viewer.spinOnce();
}






void updateViewerFromBuffersTest(std::vector<std::vector<cv::Vec3f>> &buffers, std::vector<std::vector<cv::Vec3f>>&boundary_buffer, std::vector<std::vector<float>> &vscanRes,  std::vector<cv::viz::WPolyLine> &polyLine, cv::viz::Viz3d &viewer) //std::vector<std::vector<cv::Vec3f>>boundary_buffer, 

{
    // if (buffers[0].empty()) {return;}
    viewer.removeAllWidgets();
    cv::viz::WCloudCollection collection;

    std::vector<cv::Vec3f> curbsBuffer;
    
    std::cout<<"Starting buffer loop \n"; 
    // std::cout<<"Size of buffer "<<buffers[0].size()<<"\n"; 
    // std::cout<<"Size of results "<<results[0].size()<<"\n"; 
    // for (int i = 0; i < buffers.size(); i++) 
    // {
    //     std::cout<<"i "<<i<<"\t"; 
    //     int idx = 0;
    //     for (int j = 0; j < buffers[i].size(); j++) 
    //     {
    //         std::cout<<"j "<<j<<"\t"; 
    //         if (results[i][j]) 
    //         {
    //             std::cout<<"result\t"; 
    //             curbsBuffer.push_back(buffers[i][j]);
    //         } else 
    //         {
    //             buffers[i][idx++] = buffers[i][j];
    //         }
    //         std::cout<<"Done\n"; 
    //     }
    //     std::cout<<"resizing\n"; 
    //     buffers[i].resize(idx);
    // }
    // std::cout<<"pushing \n"; 
    // buffers.push_back(curbsBuffer); 



    for (int i = 0; i < buffers.size(); i++) 
    {
        int idx = 0;
        for (int j = 0; j < buffers[i].size(); j++) 
        {
                buffers[i][idx++] = buffers[i][j];
        }
        buffers[i].resize(idx);
    }
    std::cout << "ciew pcd \n";
    for (int i = 0; i < buffers.size(); i++) 
    {
        cv::Mat cloudMat = cv::Mat(static_cast<int>(buffers[i].size()), 1, CV_32FC3, &buffers[i][0]);
        if (i == buffers.size()-1) 
        {
            collection.addCloud(cloudMat, cv::viz::Color::white());
            
        }
        else 
        {
            collection.addCloud(cloudMat, cv::viz::Color::white());
        }
    }  

    //  std::cout << "Uodated buffers \n";
    // std::cout << "trnformed l "<<boundary_buffer[0].size()<<"\n";
    // cv::Mat transformedLMat = cv::Mat(static_cast<int>(boundary_buffer[0].size()), 1, CV_32FC3, &boundary_buffer[0][0]);
    // std::cout << "trnformed l Adding\n";
    // collection.addCloud(transformedLMat, cv::viz::Color::cyan()); //Left - Cyan


    std::cout << "trnformed R \n";
    std::cout << "trnformed l "<<boundary_buffer[1].size()<<"\n";
    cv::Mat transformedRMat = cv::Mat(static_cast<int>(boundary_buffer[1].size()), 1, CV_32FC3, &boundary_buffer[1][0]);
    std::cout << "trnformed R Adding\n";
    collection.addCloud(transformedRMat, cv::viz::Color::orange()); //Right - Orange


    std::cout << "Vscan lines \n";
    int cnt = 0;
    for (auto res : vscanRes) // VScan
        {
                cv::viz::WLine line(cv::Point3f(res[0], res[1], res[2]), cv::Point3f(res[0], res[1], res[3]), cv::viz::Color::green());
                line.setRenderingProperty(cv::viz::LINE_WIDTH, 5.0); 
                viewer.showWidget("Line Widget"+std::to_string(cnt++), line);
        }

    // std::cout << "Vscan done, now boundary \n";
    // std::cout << "B1 size "<<boundary_buffer[0].size()<<"\n";
    // for (int i = 0; i < boundary_buffer[0].size(); i++){
    //     cv::Mat transformedLMat = cv::Mat(static_cast<int>(boundary_buffer[0].size()), 1, CV_32FC3, &boundary_buffer[0][0]);
    //     if (i == buffers.size()-1) 
    //     {
    //         collection.addCloud(transformedLMat, cv::viz::Color::red()); //L - purple
            
    //     }
    // }
    // std::cout << "B2 size "<<boundary_buffer[1].size()<<"\n";
    // std::cout<<"got transformedLMat \n";
    
    // for (int i = 0; i < boundary_buffer[1].size(); i++){
    //     cv::Mat transformedRMat = cv::Mat(static_cast<int>(boundary_buffer[1].size()), 1, CV_32FC3, &boundary_buffer[1][0]);
    //     if (i == buffers.size()-1) 
    //     {
    //         collection.addCloud(transformedRMat, cv::viz::Color::orange()); //R - Orange
            
    //     }
    // }
    
    std::cout<<"got transformedRMat \n";
    polyLine[0].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0); 
    polyLine[1].setRenderingProperty(cv::viz::LINE_WIDTH, 3.0);

    viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
    viewer.showWidget("Cloud", collection);
        viewer.showWidget("Poly Left", polyLine[0]);
    viewer.showWidget("Poly Right", polyLine[1]);
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
    buffer.clear();
    buffer.resize(pointcloud.size());
    for (int i = 0; i < pointcloud.size(); i++) 
    {
        buffer[i] = cv::Vec3f( pointcloud[i][0], pointcloud[i][1], pointcloud[i][2] ); 
    }
}

// void push_result_to_buffer(std::vector<cv::Vec3f> &buffer, const std::vector<std::vector<float>> &pointcloud, cv::Mat &rot, cv::Mat &trans) 
// {
//     buffer.resize(pointcloud.size());
//     for (int i = 0; i < pointcloud.size(); i++) 
//     {
//         buffer[i] = cv::Vec3f( pointcloud[i][0], pointcloud[i][1], pointcloud[i][2] ); 
//         // cv::Mat p(buffer[i]);
//         // float temp_x = p.at<float>(0,0), temp_y = p.at<float>(1,0); 
//         // buffer[i][0] = temp_y + trans.at<float>(0,0);
//         // buffer[i][1] = temp_x + trans.at<float>(1,0);
//         // buffer[i][2] = -buffer[i][2] + trans.at<float>(2,0);
//     }
// }
// }


// void push_result_to_buffer(std::vector<cv::Vec3f> &buffer, const std::vector<std::vector<float>> &pointcloud, cv::Mat &rot, cv::Mat &trans) 
// {
//     buffer.resize(pointcloud.size());
//     for (int i = 0; i < pointcloud.size(); i++) 
//     {
//         buffer[i] = cv::Vec3f( pointcloud[i][0], pointcloud[i][1], pointcloud[i][2] ); 
//         cv::Mat p(buffer[i]);
//         p = rot * p; // Rotation
//         float temp_x = p.at<float>(0,0);
//         float temp_y = p.at<float>(1,0);
//         float temp_z = p.at<float>(2,0);
//         buffer[i][0] = temp_y + trans.at<float>(0,0);
//         buffer[i][1] = temp_x + trans.at<float>(1,0);
//         //buffer[i][2] = -buffer[i][2] + trans.at<float>(2,0);
//         buffer[i][2] = -temp_z + trans.at<float>(2,0);

//         float tmp_x = buffer[i][0];
//         buffer[i][0] = buffer[i][1];
//         buffer[i][1] = tmp_x;
//         buffer[i][2] = -buffer[i][2];
//     }
// }
// }


void push_result_to_buffer_vector(std::vector<cv::Vec3f> &buffer, const std::vector<std::vector<float>> &pointcloud, cv::Mat &rot, cv::Mat &trans) 
{
    buffer.resize(pointcloud.size());
    for (int i = 0; i < pointcloud.size(); i++) 
    {
        std::vector<float> stating;
        buffer[i] = cv::Vec3f( pointcloud[i][0], pointcloud[i][1], pointcloud[i][2] ); 
        cv::Mat p(buffer[i]);
        p = rot * p; // Rotation
        float temp_x = p.at<float>(0,0);
        float temp_y = p.at<float>(1,0);
        float temp_z = p.at<float>(2,0);
        buffer[i][0] = temp_y + trans.at<float>(0,0);
        buffer[i][1] = temp_x + trans.at<float>(1,0);
        //buffer[i][2] = -buffer[i][2] + trans.at<float>(2,0);
        buffer[i][2] = -temp_z + trans.at<float>(2,0);

        float tmp_x = buffer[i][0];
        buffer[i][0] = buffer[i][1];
        buffer[i][1] = tmp_x;
        buffer[i][2] = -buffer[i][2];   
    }
    
    for(int i=0; i<buffer.size();i++){
        float *p = (float*)pointcloud[i].data();
        p[0] = buffer[i][0];
        p[1] = buffer[i][1];
        p[2] = buffer[i][2]; 
    }
}
}
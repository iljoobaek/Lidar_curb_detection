#include "boundary_detection.h"
#include "viewer.h"
#include "sensor_config.h"

#include "rosbag/bag.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/point_cloud2_iterator.h"
#include "std_msgs/String.h"
#include "unistd.h"

std::vector<ros::Time> timeStamps;

void capture_and_detect(const std::unique_ptr<Boundary_detection> &detection, std::vector<cv::Vec3f> &buffer, std::vector<int> &result, float theta, cv::Mat &rot, cv::Mat &trans) 
{ 
    std::vector<velodyne::Laser> laser;
    // Capture one frame
    *(detection->capture) >> laser;
    // Convert pointcloud to cartesian and copy to detection object 
    LidarViewer::laser_to_cartesian(laser, detection->get_pointcloud(), theta, rot, trans);
    // Run detection
    detection->detect(rot, trans, false);
    // Push result to buffer 
    LidarViewer::push_result_to_buffer(buffer, detection->get_pointcloud(), rot, trans);
    std::copy(detection->get_result().begin(), detection->get_result().end(), std::back_inserter(result));
}

void capture_and_detect_bool(const std::unique_ptr<Boundary_detection> &detection, std::vector<cv::Vec3f> &buffer, std::vector<bool> &result, float theta, cv::Mat &rot, cv::Mat &trans) 
{ 
    std::vector<velodyne::Laser> laser;
    // Capture one frame
    *(detection->capture) >> laser;
    // Convert pointcloud to cartesian and copy to detection object 
    LidarViewer::laser_to_cartesian(laser, detection->get_pointcloud(), theta, rot, trans);
    // Run detection
    detection->detect(rot, trans, false);
    // Push result to buffer 
    LidarViewer::push_result_to_buffer(buffer, detection->get_pointcloud(), rot, trans);
    std::copy(detection->get_result_bool().begin(), detection->get_result_bool().end(), std::back_inserter(result));
}

void writeMsgToRosbag(rosbag::Bag &bag, const std::vector<std::vector<cv::Vec3f>> &buffers, int lidarIdx, int frameIdx) 
{ 
    int numOfPoints = 0;
    for (auto &buffer : buffers) numOfPoints += buffer.size();
    
    sensor_msgs::PointCloud2 msg;
    msg.header.frame_id = "velodyne";
    msg.height = 1;
    msg.width = numOfPoints;
    
    sensor_msgs::PointCloud2Modifier modifier(msg);
    modifier.setPointCloud2Fields(4, "x", 1, sensor_msgs::PointField::FLOAT32,
                                     "y", 1, sensor_msgs::PointField::FLOAT32,
                                     "z", 1, sensor_msgs::PointField::FLOAT32,
                                     "intensity", 1, sensor_msgs::PointField::FLOAT32);
    modifier.resize(numOfPoints);

    sensor_msgs::PointCloud2Iterator<float> px(msg, "x");
    sensor_msgs::PointCloud2Iterator<float> py(msg, "y");
    sensor_msgs::PointCloud2Iterator<float> pz(msg, "z");
    sensor_msgs::PointCloud2Iterator<float> pi(msg, "intensity");
    // sensor_msgs::PointCloud2Iterator<float> pr(msg, "r");
    
    for (auto &buffer : buffers) 
    {
        for (auto &point : buffer) 
        {
            *px = point[0];
            *py = point[1];
            *pz = point[2];
            *pi = point[3];
            // *pr = point[2];

            ++px;
            ++py;
            ++pz;
            ++pi;
        }
    }
    if (lidarIdx == 0) {
        ros::Time timestamp = ros::Time::now();
        msg.header.stamp = timestamp;
        // bag.write("/velodyne_points", timestamp, msg);
        bag.write("/points_raw", timestamp, msg);
        timeStamps.push_back(timestamp);
    }
    else {
        msg.header.stamp = timeStamps[frameIdx];
        // bag.write("/velodyne_points", timeStamps[frameIdx], msg);
        bag.write("/points_raw", timeStamps[frameIdx], msg);
    }
}

int main(int argc, char* argv[]) 
{
    int numOfVelodynes = 6; 
    double total_ms = 0.0;

    std::vector<std::string> pcap_files = SensorConfig::getPcapFiles(); 
    std::vector<std::string> fn_out;
    for (auto &fn : pcap_files)
    {
        fn_out.push_back(fn.substr(0, fn.find('.')) + ".bag");
        std::cout << fn_out.back() << std::endl;
    }

    // Rosbag option
    ros::Time::init();
    
    // // Create Viz3d Viewer and register callbacks
    // cv::viz::Viz3d viewer( "Velodyne" );
    // bool pause(false);
    // LidarViewer::cvViz3dCallbackSetting(viewer, pause);
    
    // Get rotation and translation parameters from lidar to vehicle coordinate
    auto rot_params = SensorConfig::getRotationParams();
    auto rot_vec = SensorConfig::getRotationMatrices(rot_params);
    auto trans_vec = SensorConfig::getTranslationMatrices();

    std::vector<std::unique_ptr<Boundary_detection>> detections;
    for (auto &file : pcap_files) 
    {
        detections.push_back(std::unique_ptr<Boundary_detection> (new Boundary_detection(file, 0., 1.125)));
    }

    for (int i = 0; i < numOfVelodynes; i++)
    {
        rosbag::Bag bag_out(fn_out[i], rosbag::bagmode::Write);
        int frame_idx = 0;
        while (detections[i]->capture->isRun()) 
        {
            auto t_start = std::chrono::system_clock::now();
            
            std::vector<std::vector<cv::Vec3f>> buffers(1); 
            std::vector<bool> result; 
            
            capture_and_detect_bool(detections[i], buffers[0], result, rot_params[i], rot_vec[i], trans_vec[i]);
            // LidarViewer::updateViewerFromBuffers(buffers, results, viewer);

            // Save message to rosbag 
            writeMsgToRosbag(bag_out, buffers, i, frame_idx); 

            auto t_end = std::chrono::system_clock::now();
            std::chrono::duration<double, std::milli> fp_ms = t_end - t_start;
            std::cout << "Frame " << frame_idx++ << ": takes " << fp_ms.count() << " ms" << std::endl;
            usleep(80000);
            if (frame_idx > 150) break;
        }
        bag_out.close();
    }

    // int frame_idx = 0;
    // while (detections[0]->capture->isRun()) 
    // {
    //     auto t_start = std::chrono::system_clock::now();
        
    //     std::vector<std::vector<cv::Vec3f>> buffers(numOfVelodynes); 
    //     std::vector<std::vector<bool>> results(numOfVelodynes); 
    //     std::vector<std::vector<int>> results_int(numOfVelodynes); 
        
    //     // Read in one frame and run detection
    //     std::thread th[numOfVelodynes];
    //     for (int i = 0; i < numOfVelodynes; i++) 
    //     {
    //         // Convert to 3-dimention Coordinates
    //         th[i] = std::thread(capture_and_detect_bool, std::ref(detections[i]), std::ref(buffers[i]), std::ref(results[i]), rot_params[i], std::ref(rot_vec[i]), std::ref(trans_vec[i]));
    //     }
    //     for (int i = 0; i < numOfVelodynes; i++) 
    //     {
    //         th[i].join();
    //     }
    //     // LidarViewer::updateViewerFromBuffers(buffers, results, viewer);

    //     // Save message to rosbag 
    //     writeMsgToRosbag(bag_out, buffers); 

    //     auto t_end = std::chrono::system_clock::now();
    //     std::chrono::duration<double, std::milli> fp_ms = t_end - t_start;
    //     std::cout << "Frame " << frame_idx++ << ": takes " << fp_ms.count() << " ms" << std::endl;
    // }
    // bag_out.close();
    return 0;
}

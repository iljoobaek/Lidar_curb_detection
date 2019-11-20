#include "boundary_detection.h"
#include "viewer.h"
#include "sensor_config.h"

#include "rosbag/bag.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/point_cloud2_iterator.h"
#include "std_msgs/String.h"

#include "fastvirtualscan/fastvirtualscan.h"

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

void writeMsgToRosbag(rosbag::Bag &bag, const std::vector<std::vector<cv::Vec3f>> &buffers) 
{ 
    int numOfPoints = 0;
    for (auto &buffer : buffers) numOfPoints += buffer.size();
    
    sensor_msgs::PointCloud2 msg;
    msg.header.frame_id = "/velodyne";
    msg.height = 1;
    msg.width = numOfPoints;
    
    sensor_msgs::PointCloud2Modifier modifier(msg);
    modifier.setPointCloud2Fields(3, "x", 1, sensor_msgs::PointField::FLOAT32,
                                     "y", 1, sensor_msgs::PointField::FLOAT32,
                                     "z", 1, sensor_msgs::PointField::FLOAT32);
    modifier.resize(numOfPoints);

    sensor_msgs::PointCloud2Iterator<float> px(msg, "x");
    sensor_msgs::PointCloud2Iterator<float> py(msg, "y");
    sensor_msgs::PointCloud2Iterator<float> pz(msg, "z");
    // sensor_msgs::PointCloud2Iterator<float> pi(msg, "i");
    // sensor_msgs::PointCloud2Iterator<float> pr(msg, "r");
    
    for (auto &buffer : buffers) 
    {
        for (auto &point : buffer) 
        {
            *px = point[0];
            *py = point[1];
            *pz = point[2];
            // *pi = point[1];
            // *pr = point[2];

            ++px;
            ++py;
            ++pz;
        }
    }
    ros::Time timestamp = ros::Time::now();
    bag.write("/points_raw", timestamp, msg);
}

// Rosbag option
// ros::Time::init();
// rosbag::Bag bag_out("six_lidars(front).bag", rosbag::bagmode::Write);

int main(int argc, char* argv[]) 
{
    // Number of velodyne sensors, maximum 6
    int numOfVelodynes;
    if (argc < 2)
    {
        numOfVelodynes = 6;
    } 
    else if (argc == 2) 
    {
        numOfVelodynes = std::stoi(argv[1]);
        if (numOfVelodynes < 1 || numOfVelodynes > 6)
        {
            std::cerr << "Invalid number of Velodynes, should be from 1 to 6.\n";
            return -1;
        }
    }
    else {
        std::cerr << "Invalid arguments.\n";
        return -1;
    }
    std::cout << "numOfVelodynes: " << numOfVelodynes << "\n";
    
    std::vector<std::string> pcap_files = SensorConfig::getPcapFiles(); 
    
    // Create Viz3d Viewer and register callbacks
    cv::viz::Viz3d viewer( "Velodyne" );
    bool pause(false);
    LidarViewer::cvViz3dCallbackSetting(viewer, pause);
    
    // Get rotation and translation parameters from lidar to vehicle coordinate
    auto rot_params = SensorConfig::getRotationParams();
    auto rot_vec = SensorConfig::getRotationMatrices(rot_params);
    auto trans_vec = SensorConfig::getTranslationMatrices();

    std::vector<std::unique_ptr<Boundary_detection>> detections;
    for (auto &file : pcap_files) 
    {
        detections.push_back(std::unique_ptr<Boundary_detection> (new Boundary_detection(file, 0., 1.125)));
    }

    // Virtual scan object
    FastVirtualScan virtualscan = FastVirtualScan();

    // Main loop
    int frame_idx = 0;
    while (detections[0]->capture->isRun() && !viewer.wasStopped()) 
    {
        if (pause) 
        {
            viewer.spinOnce();
            continue;
        }
        auto t_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        
        std::vector<std::vector<cv::Vec3f>> buffers(numOfVelodynes); 
        std::vector<std::vector<bool>> results(numOfVelodynes); 
        std::vector<std::vector<int>> results_int(numOfVelodynes); 
        
        // Read in one frame and run detection
        std::thread th[numOfVelodynes];
        for (int i = 0; i < numOfVelodynes; i++) 
        {
            // Convert to 3-dimention Coordinates
            th[i] = std::thread(capture_and_detect_bool, std::ref(detections[i]), std::ref(buffers[i]), std::ref(results[i]), rot_params[i], std::ref(rot_vec[i]), std::ref(trans_vec[i]));
        }
        for (int i = 0; i < numOfVelodynes; i++) 
        {
            th[i].join();
        }
        LidarViewer::updateViewerFromBuffers(buffers, results, viewer);
        
        // Run virtualscan algorithm 
        // virtualscan.calculateVirtualScans(BEAMNUM, STEP, MINFLOOR, MAXCEILING, OBSTACLEMINHEIGHT, MAXBACKDISTANCE, 
        //                                   ROTATION * PI / 180.0, MINRANGE);

        // virtualscan.getVirtualScan(ROADSLOPMINHEIGHT * PI / 180.0, ROADSLOPMAXHEIGHT * PI / 180.0, MAXFLOOR, MINCEILING, 
        //                            PASSHEIGHT, beams);


        auto t_end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t_start;
        std::cout << "Frame " << frame_idx++ << ": takes " << t_end << " ms" << std::endl;
    }
    viewer.close();
    return 0;
}

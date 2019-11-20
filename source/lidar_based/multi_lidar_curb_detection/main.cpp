#include "boundary_detection.h"
#include "viewer.h"
#include "rosbag/bag.h"
#include "sensor_msgs/PointCloud2.h"
#include "sensor_msgs/point_cloud2_iterator.h"
#include "std_msgs/String.h"

void capture_and_detect(const std::unique_ptr<Boundary_detection> &detection, std::vector<cv::Vec3f> &buffer, std::vector<int> &result, float theta, cv::Mat &rot, cv::Mat &trans) { 
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

void capture_and_detect_bool(const std::unique_ptr<Boundary_detection> &detection, std::vector<cv::Vec3f> &buffer, std::vector<bool> &result, float theta, cv::Mat &rot, cv::Mat &trans) { 
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

void write_to_rosbag(rosbag::Bag &bag, const std::vector<std::vector<cv::Vec3f>> &buffers) { 
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
    
    for (auto &buffer : buffers) {
        for (auto &point : buffer) {
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

int main( int argc, char* argv[] ) {
    // Number of velodyne sensors, maximum 6 
    int numOfVelodynes = 1;
    int lidar_idx = 5;
    std::vector<std::string> pcap_files = LidarViewer::get_file_names(); 
    
    // Create Viewer
    cv::viz::Viz3d viewer( "Velodyne" );
    bool pause(false);

    // Register Keyboard Callback
    viewer.registerKeyboardCallback(
        []( const cv::viz::KeyboardEvent& event, void* cookie ){
        // Close Viewer
        if( event.code == 'q' && event.action == cv::viz::KeyboardEvent::Action::KEY_DOWN ){
            static_cast<cv::viz::Viz3d*>( cookie )->close();
        }
        }
        , &viewer);
    viewer.registerKeyboardCallback(
        []( const cv::viz::KeyboardEvent& event, void* pause ){
        // Switch state of pause / resume when pressing p
        if( event.code == 'p' && event.action == cv::viz::KeyboardEvent::Action::KEY_DOWN ){
            bool* p = static_cast<bool*>( pause );
            *p = !(*p);
        }
        }
        , &pause);
    
    // Get rotation and translation parameters from lidar to vehicle coordinate
    auto rot_params = LidarViewer::get_rot_params();
    auto rot_vec = LidarViewer::get_rotation_matrices(rot_params);
    auto trans_vec = LidarViewer::get_translation_matrices();
    
    std::vector<std::unique_ptr<Boundary_detection>> detections;
    for (auto &file : pcap_files) {
        detections.push_back(std::unique_ptr<Boundary_detection> (new Boundary_detection(file, 0, 0., 1.125)));
    }

    // Rosbag option
    ros::Time::init();
    rosbag::Bag bag_out("six_lidars(front).bag", rosbag::bagmode::Write);

    // Main loop
    int frame_idx = 0;
    while (detections[0]->capture->isRun() && !viewer.wasStopped()) {
    // while (detections[lidar_idx]->capture->isRun() && !viewer.wasStopped()) {
        if (pause) {
            viewer.spinOnce();
            continue;
        }
        auto t_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        
        std::vector<std::vector<cv::Vec3f>> buffers(numOfVelodynes); 
        std::vector<std::vector<bool>> results(numOfVelodynes); 
        std::vector<std::vector<int>> results_int(numOfVelodynes); 
        
        // Read in one frame and run detection
        std::thread th[numOfVelodynes];
        for (int i = 0; i < numOfVelodynes; i++) {
        // for (int i = lidar_idx; i < lidar_idx+1; i++) {
            // Convert to 3-dimention Coordinates
            th[i] = std::thread(capture_and_detect_bool, std::ref(detections[i]), std::ref(buffers[i]), std::ref(results[i]), rot_params[i], std::ref(rot_vec[i]), std::ref(trans_vec[i]));
            // th[i] = std::thread(capture_and_detect, std::ref(detections[i]), std::ref(buffers[i]), std::ref(results_int[i]), rot_params[i], std::ref(rot_vec[i]), std::ref(trans_vec[i]));
        }
        for (int i = 0; i < numOfVelodynes; i++) {
        // for (int i = lidar_idx; i < lidar_idx+1; i++) {
            th[i].join();
        }
        LidarViewer::update_viewer(buffers, results, viewer);
        // LidarViewer::update_viewer(buffers, results_int, viewer);
        write_to_rosbag(bag_out, buffers);
        auto t_end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t_start;
        std::cout << "Frame " << frame_idx++ << ": takes " << t_end << " ms" << std::endl;
        for (int i = 0; i < numOfVelodynes; i++) {
            std::cout << detections[i]->get_pointcloud().size() << " ";
        }
        std::cout << std::endl;
    }
    bag_out.close();
    viewer.close();
    return 0;
}

#include "boundary_detection.h"
#include "viewer.h"

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

int main( int argc, char* argv[] ) {
    
    int numOfVelodynes = 1;
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
        // Close Viewer
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

    int frame_idx = 0;
    while (detections[0]->capture->isRun() && !viewer.wasStopped()) {
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
            // Convert to 3-dimention Coordinates
            th[i] = std::thread(capture_and_detect_bool, std::ref(detections[i]), std::ref(buffers[i]), std::ref(results[i]), rot_params[i], std::ref(rot_vec[i]), std::ref(trans_vec[i]));
            // th[i] = std::thread(capture_and_detect, std::ref(detections[i]), std::ref(buffers[i]), std::ref(results_int[i]), rot_params[i], std::ref(rot_vec[i]), std::ref(trans_vec[i]));
        }
        for (int i = 0; i < numOfVelodynes; i++) {
            th[i].join();
        }
        LidarViewer::update_viewer(buffers, results, viewer);
        // LidarViewer::update_viewer(buffers, results_int, viewer);
        auto t_end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t_start;
        std::cout << "Frame " << frame_idx++ << ": takes " << t_end << " ms" << std::endl;
        for (int i = 0; i < numOfVelodynes; i++) {
            std::cout << detections[i]->get_pointcloud().size() << " ";
        }
        std::cout << std::endl;
    }

    viewer.close();
    return 0;
}

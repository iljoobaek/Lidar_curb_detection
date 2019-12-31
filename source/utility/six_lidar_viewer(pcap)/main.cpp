#include <csignal>

#include "viewer.h"
#include "sensor_config.h"
#include "VelodyneCapture.h"

#include "fastvirtualscan/fastvirtualscan.h"

static volatile sig_atomic_t sig_caught = 0;

static int BEAMNUM = 2000;
static double STEP = 0.05;
static double MINFLOOR = -2.0;
static double MAXFLOOR = 1.0;
static double MAXCEILING = 6.0;
static double MINCEILING = -0.5;
static double ROADSLOPMINHEIGHT = 80.0;
static double ROADSLOPMAXHEIGHT = 30.0;
static double ROTATION = 3.0;
static double OBSTACLEMINHEIGHT = 1.0;
static double MAXBACKDISTANCE = 1.0;
static double PASSHEIGHT = 3.0;

static double MAXRANGE = 20.0;
static double MINRANGE = 2.0;
static double GRIDSIZE = 10.0;
static double IMAGESIZE = 1000.0;

std::vector<std::vector<float>> getVscanResult(const FastVirtualScan &virtualscan, const QVector<double> &beams)
{
    std::vector<std::vector<float>> res;
    double density = 2 * CV_PI / BEAMNUM;
    for (int i = 0; i < BEAMNUM; i++)
    {
        double theta = i * density - CV_PI;
        if (beams[i] == 0 || virtualscan.minheights[i] == virtualscan.maxheights[i])
        {
            continue;
        }
        float x = beams[i] * std::cos(theta);
        float y = beams[i] * std::sin(theta);
        float minHeight = virtualscan.minheights[i];
        float maxHeight = virtualscan.maxheights[i];
        res.push_back({x, y, minHeight, maxHeight});
    }  
    return res; 
}

std::vector<cv::Vec3f> vectorToVec3f(const std::vector<std::vector<float>> &vec)
{
    std::vector<cv::Vec3f> res;
    for (auto &v : vec)
    {
        res.push_back(cv::Vec3f(v[0], v[1], 0.0f));
    }
    return res;
}

void capture_and_detect_bool(velodyne::VLP16Capture *capture, std::vector<cv::Vec3f> &buffer, std::vector<bool> &result, float theta, cv::Mat &rot, cv::Mat &trans) { 
    std::vector<velodyne::Laser> laser;
    std::vector<std::vector<float>> pointcloud;
    // Capture one frame
    (*capture) >> laser;
    // Convert pointcloud to cartesian and copy to detection object 
    LidarViewer::laser_to_cartesian(laser, pointcloud, theta, rot, trans);
    // Push result to buffer 
    LidarViewer::push_result_to_buffer(buffer, pointcloud, rot, trans);
    result = std::vector<bool>(pointcloud.size(), false);
}

void signalHandler(int signum)
{
    sig_caught = 1;
}

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

    // Signal Handler for pause/resume viewer
    std::signal(SIGINT, signalHandler);
    double total_ms = 0.0;

    // // Get pcap file names
    std::vector<std::string> pcap_files = SensorConfig::getPcapFiles(); 
    
    // Create Viz3d Viewer and register callbacks
    cv::viz::Viz3d viewer( "Velodyne" );
    bool pause(false);
    LidarViewer::cvViz3dCallbackSetting(viewer, pause);
    
    // Get rotation and translation parameters from lidar to vehicle coordinate
    auto rot_params = SensorConfig::getRotationParams();
    auto rot_vec = SensorConfig::getRotationMatrices(rot_params);
    auto trans_vec = SensorConfig::getTranslationMatrices();

    // Boundary detection object
    std::vector<velodyne::VLP16Capture*> captures;
    for (auto file : pcap_files) 
    {
        captures.push_back(new velodyne::VLP16Capture(file));
    }

    // Virtual scan object
    FastVirtualScan virtualscan = FastVirtualScan();

    // Main loop
    int frame_idx = 0;
    while (captures[0]->isRun() && !viewer.wasStopped()) 
    {
        if (sig_caught)
        {
            std::cout << "Average time per frame: " << (total_ms / (frame_idx-10)) << " ms\n";
            return -1;
        }
        if (pause) 
        {
            viewer.spinOnce();
            continue;
        }
        
        std::vector<std::vector<cv::Vec3f>> buffers(numOfVelodynes); 
        std::vector<std::vector<bool>> results(numOfVelodynes); 
        std::vector<cv::Vec3f> bufferAll; 

        int lidar_number = 3;

        auto t_start = std::chrono::system_clock::now();
        // Read in one frame and run detection
        std::thread th[numOfVelodynes];
        for (int i = 0; i < numOfVelodynes; i++) 
        {
            // Convert to 3-dimention Coordinates
            th[i] = std::thread(capture_and_detect_bool, std::ref(captures[i]), std::ref(buffers[i]), std::ref(results[i]), rot_params[i], std::ref(rot_vec[i]), std::ref(trans_vec[i]));
        }
        for (int i = 0; i < numOfVelodynes; i++) 
        {
            th[i].join();
        }
        for (auto &buffer : buffers)
        {
            std::copy(buffer.begin(), buffer.end(), std::back_inserter(bufferAll));
        }
        // Run virtualscan algorithm 
        QVector<double> beams; 
        virtualscan.calculateVirtualScans(bufferAll, BEAMNUM, STEP, MINFLOOR, MAXCEILING, OBSTACLEMINHEIGHT, MAXBACKDISTANCE, 
                                          ROTATION * CV_PI / 180.0, MINRANGE);

        virtualscan.getVirtualScan(ROADSLOPMINHEIGHT * CV_PI / 180.0, ROADSLOPMAXHEIGHT * CV_PI / 180.0, MAXFLOOR, MINCEILING, 
                                   PASSHEIGHT, beams);

        auto res = getVscanResult(virtualscan, beams);

        LidarViewer::updateViewerFromBuffers(buffers, results, viewer, res, {});
        auto t_end = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t_end - t_start;
        std::cout << "Frame " << frame_idx++ << ": takes " << fp_ms.count() << " ms" << std::endl;
        if (frame_idx >= 10)
        {
            total_ms += fp_ms.count();
        }
    }
    viewer.close();
    std::cout << "Average time per frame: " << (total_ms / (frame_idx-10)) << " ms\n";
    return 0;
}

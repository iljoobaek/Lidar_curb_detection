#include <csignal>

#include "boundary_detection.h"
#include "viewer.h"
#include "sensor_config.h"

#include "fastvirtualscan/fastvirtualscan.h"

// static int BEAMNUM = 1440;
static int BEAMNUM = 720;
static double STEP = 0.05;
static double MINFLOOR = -2.0;
static double MAXFLOOR = -1.0;
static double MAXCEILING = 6.0;
static double MINCEILING = -0.5;
static double ROADSLOPMINHEIGHT = 80.0;
static double ROADSLOPMAXHEIGHT = 30.0;
static double ROTATION = 3.0;
static double OBSTACLEMINHEIGHT = 1.0;
static double MAXBACKDISTANCE = 1.0;
static double PASSHEIGHT = 2.0;

static double MAXRANGE = 20.0;
static double MINRANGE = 2.0;
static double GRIDSIZE = 10.0;
static double IMAGESIZE = 1000.0;

static volatile sig_atomic_t sig_caught = 0;

void signalHandler(int signum)
{
    sig_caught = 1;
}

std::vector<std::vector<float>> getVscanResult(const FastVirtualScan &virtualscan, const QVector<double> &beams)
{
    std::vector<std::vector<float>> res;
    double density = 2 * PI / BEAMNUM;
    for (int i = 0; i < BEAMNUM; i++)
    {
        double theta = i * density - PI;
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

void capture_and_detect(const std::unique_ptr<Boundary_detection> &detection, std::vector<cv::Vec3f> &buffer, std::vector<int> &result, float theta, cv::Mat &rot, cv::Mat &trans) 
{ 
    std::vector<velodyne::Laser> laser;
    // Capture one frame
    *(detection->capture) >> laser;
    // Convert pointcloud to cartesian and copy to detection object 
    DataSource::laser_to_cartesian(laser, detection->get_pointcloud(), theta, rot, trans);
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
    DataSource::laser_to_cartesian(laser, detection->get_pointcloud(), theta, rot, trans);
    // Run detection
    detection->detect(rot, trans, false);
    // Push result to buffer 
    LidarViewer::push_result_to_buffer(buffer, detection->get_pointcloud(), rot, trans);
    std::copy(detection->get_result_bool().begin(), detection->get_result_bool().end(), std::back_inserter(result));
}

void captureFromBinaryFiles(int frame_idx, std::vector<cv::Vec3f> &buffer, std::vector<bool> &result, float theta, float sensor_height) 
{ 
    std::string fn = SensorConfig::getBinaryFile(frame_idx);
    std::vector<std::vector<float>> pointcloud = DataSource::readFromBinary(fn);
    DataSource::rotateAndTranslate(pointcloud, theta, sensor_height);
    LidarViewer::pushToBuffer(buffer, pointcloud);
    result = std::vector<bool>(pointcloud.size(), false);
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
    numOfVelodynes = 1;

    std::signal(SIGINT, signalHandler);
    double total_ms = 0.0;

    DataReader::LidarDataReader reader("20191126163620/", 0, 1324);

    // Get pcap file names
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
        detections.push_back(std::unique_ptr<Boundary_detection> (new Boundary_detection(0., 1.125)));
    }

    // Virtual scan object
    FastVirtualScan virtualscan = FastVirtualScan();
    fusion::FusionController fuser;

    // Main loop
    int frame_idx = 0;
    while (reader.isRun() && !viewer.wasStopped()) 
    // while (detections[0]->capture->isRun() && !viewer.wasStopped()) 
    // for (int idx = 1750; idx < 2250; idx++) 
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
        std::vector<std::vector<int>> results_int(numOfVelodynes); 
        
        // // Read in one frame and run detection
        // std::thread th[numOfVelodynes];
        // for (int i = 0; i < numOfVelodynes; i++) 
        // {
        //     // Convert to 3-dimention Coordinates
        //     th[i] = std::thread(capture_and_detect_bool, std::ref(detections[i]), std::ref(buffers[i]), std::ref(results[i]), rot_params[i], std::ref(rot_vec[i]), std::ref(trans_vec[i]));
        // }
        // for (int i = 0; i < numOfVelodynes; i++) 
        // {
        //     th[i].join();
        // }
        
        // captureFromBinaryFiles(idx, buffers[0], results[0], 16.0f, 1.125f); 
        
        // auto pointcloud = detections[0]->get_pointcloud();
        // reader >> pointcloud;
        reader >> detections[0]->get_pointcloud();
        detections[0]->detect(rot_vec[0], trans_vec[0]);
        LidarViewer::pushToBuffer(buffers[0], detections[0]->get_pointcloud());
        results[0] = std::vector<bool>(detections[0]->get_pointcloud().size(), false);
        
        auto t_start = std::chrono::system_clock::now();
        // Run virtualscan algorithm 
        QVector<double> beams; 
        virtualscan.calculateVirtualScans(buffers[0], BEAMNUM, STEP, MINFLOOR, MAXCEILING, OBSTACLEMINHEIGHT, MAXBACKDISTANCE, 
                                          ROTATION * PI / 180.0, MINRANGE);

        virtualscan.getVirtualScan(ROADSLOPMINHEIGHT * PI / 180.0, ROADSLOPMAXHEIGHT * PI / 180.0, MAXFLOOR, MINCEILING, 
                                   PASSHEIGHT, beams);

        auto res = getVscanResult(virtualscan, beams);
        // auto buf = vectorToVec3f(res);
        auto t_end = std::chrono::system_clock::now();

        auto buf = detections[0]->getLidarBuffers(detections[0]->get_pointcloud(), detections[0]->get_result_bool());
        std::vector<cv::viz::WPolyLine> thirdOrder = fuser.displayThirdOrder(buf[1]);

        results[0] = detections[0]->get_result_bool();
        LidarViewer::updateViewerFromBuffers(buffers, results, viewer, res, thirdOrder);

        // auto t_end = std::chrono::system_clock::now();
        std::chrono::duration<double, std::milli> fp_ms = t_end - t_start;
        std::cout << "Frame " << frame_idx++ << ": takes " << fp_ms.count() << " ms for vscan" << std::endl;
        if (frame_idx >= 10)
        {
            total_ms += fp_ms.count();
        }
    }
    viewer.close();
    std::cout << "Average time per frame: " << (total_ms / (frame_idx-10)) << " ms\n";

    return 0;
}

#include <csignal>

#include "timers.h"
#include <ctime>
#include <chrono>
#include <stdlib.h>
#include <time.h>

#include "viewer.h"
#include "sensor_config.h"

#include "fastvirtualscan/fastvirtualscan.h"
#include "data_reader.h"

#include <omp.h>

#include <iostream>
// # include <Python.h>
// #include <Boost/Python.h>
#include <Python.h>
#include <python2.7/Python.h>
#include <boost/python.hpp>
//#include "/home/rtml/.local/lib/python2.7/site-packages/numpy/core/include/numpy/arrayobject.h"
// #include "/home/rtml/Kamal_Workspace/Lidar_curb_detection/source/vscan_based/env/lib/python2.7/site-packages/numpy/core/include/numpy/arrayobject.h"
#include <exception>
#include <fstream>
#include <string>
#include <sstream>

Timers timers = Timers();
extern std::string optType;
extern std::string optInfo;

// Parameters for virtual scan
#define PI 3.14159265
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

int main(int argc, char *argv[])
{
    setupTimers(timers);
    // Number of velodyne sensors, maximum 6
    int numOfVelodynes;
    std::string processNum;
    if (argc < 2)
    {
        numOfVelodynes = 6;
    }
    else if (argc >= 3)
    {
        numOfVelodynes = std::stoi(argv[1]);
        if (numOfVelodynes < 1 || numOfVelodynes > 6)
        {
            std::cerr << "Invalid number of Velodynes, should be from 1 to 6.\n";
            return -1;
        }
    }
    else
    {
        std::cerr << "Invalid arguments.\n";
        return -1;
    }
    numOfVelodynes = 1;

    // Signal Handler for pause/resume viewer
    std::signal(SIGINT, signalHandler);
    double total_ms = 0.0;

    // // Get pcap file names
    // std::vector<std::string> pcap_files = SensorConfig::getPcapFiles();

    // Create Viz3d Viewer and register callbacks
    cv::viz::Viz3d viewer("Velodyne");
    bool pause(false);
    LidarViewer::cvViz3dCallbackSetting(viewer, pause);

    // Get rotation and translation parameters from lidar to vehicle coordinate
    auto rot_params = SensorConfig::getRotationParams();
    auto rot_vec = SensorConfig::getRotationMatrices(rot_params);
    auto trans_vec = SensorConfig::getTranslationMatrices();

    int frameStart = 0, frameEnd = 153;

    int numFrames = 1 + frameEnd - frameStart;

    // Virtual scan object
    FastVirtualScan virtualscan = FastVirtualScan(BEAMNUM, STEP, MINFLOOR, MAXCEILING);
    // FastVirtualScan virtualscan = FastVirtualScan(1000, 0.3, -3, 3);

    std::vector<std::vector<float>> pointcloud;
    std::string root_path = "/home/tarang/Lidar_Project_Fall_2019_Tarang/data/kitti_data/"; // Ensure path has string "kitti" inside it if this is Kitti data
    std::string data_folder = "2011_09_26_drive_0005_sync/";                                //2011_09_26_drive_0005_sync  2011_09_26_drive_0051_sync
    bool is_kitti_downsample = true;                                                        // This should be true if using Kitti Data
    DataReader::LidarDataReader dataReader(root_path, data_folder, frameStart, frameEnd + 1, is_kitti_downsample);

    // Main loop
    int frame_idx = frameStart;
    while (dataReader.isRun() && !viewer.wasStopped())
    {

        timers.resetTimer("frame");
        if (sig_caught)
        {
            //std::cout << "Average time per frame: " << (total_ms / (frame_idx-10)) << " ms\n";
            return -1;
        }
        if (pause)
        {
            viewer.spinOnce();
            continue;
        }

        std::vector<std::vector<cv::Vec3f>> buffers(numOfVelodynes);

        dataReader >> pointcloud;

        LidarViewer::pushToBuffer(buffers[0], pointcloud);

        auto t_start = std::chrono::system_clock::now();

        timers.resetTimer("virtualscan");

        // Run virtualscan algorithm
        QVector<double> beams;

        timers.resetTimer("calculateVirtualScans");
        virtualscan.calculateVirtualScans(buffers[0], BEAMNUM, STEP, MINFLOOR, MAXCEILING, OBSTACLEMINHEIGHT, MAXBACKDISTANCE, ROTATION * PI / 180.0, MINRANGE);
        timers.pauseTimer("calculateVirtualScans");

        timers.resetTimer("getVirtualScan");
        virtualscan.getVirtualScan(ROADSLOPMINHEIGHT * PI / 180.0, ROADSLOPMAXHEIGHT * PI / 180.0, MAXFLOOR, MINCEILING,
                                   PASSHEIGHT, beams);
        timers.pauseTimer("getVirtualScan");

        timers.resetTimer("getVscanResult");
        auto res = getVscanResult(virtualscan, beams);
        timers.pauseTimer("getVscanResult");

        timers.pauseTimer("virtualscan");

        auto t_end = std::chrono::system_clock::now();

        // visualization
        timers.resetTimer("visualization");
        LidarViewer::updateViewerFromBuffers(buffers, viewer, res);
        timers.pauseTimer("visualization");

        std::chrono::duration<double, std::milli> fp_ms = t_end - t_start;
        std::cout << "Frame " << frame_idx++ << ": takes " << fp_ms.count() << " ms for vscan" << std::endl;
        if (frame_idx >= 10)
        {
            total_ms += fp_ms.count();
        }

        timers.pauseTimer("frame");
    } // End of main while loop

    viewer.close();
    std::cout << "Average time per frame: " << (total_ms / (frame_idx - 10)) << " ms\n";

    // Timers Stuff. Written to timersLog.txt
    auto now = std::chrono::system_clock::now();
    auto t_c = std::chrono::system_clock::to_time_t(now - std::chrono::hours(24));

    /*std::stringstream ss;
	ss << std::put_time(std::localtime(&t_c), "%F_%X");
	std::string timerLogFileName = "timersLog/" + optType + "_" + ss.str() + "_" + processNum + ".txt";
	
	std::cout << timerLogFileName << std::endl;
    timers.printToFile(timerLogFileName, optType, optInfo);*/

    timers.print();

    return 0;
}

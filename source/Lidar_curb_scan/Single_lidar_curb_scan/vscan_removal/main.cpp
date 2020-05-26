#include <csignal>

#include "boundary_detection.h"
#include "viewer.h"
#include "sensor_config.h"

#include "fastvirtualscan/fastvirtualscan.h"

// Parameters for virtual scan
static int BEAMNUM = 720;
static double STEP = 0.5;
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

std::pair<std::vector<std::vector<float>>, std::unordered_map<int, int>> getVscanResult(const FastVirtualScan &virtualscan, const QVector<double> &beams)
{
    std::vector<std::vector<float>> res;
    std::unordered_map<int, int> m;
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
        float thetaT = std::atan2(y, x);
        if (thetaT < 0.0f) 
        {
            thetaT += 2 * PI;
        }
        int ithBeam = thetaT / density;
        float minHeight = virtualscan.minheights[i];
        float maxHeight = virtualscan.maxheights[i];
        float dist = std::sqrt(x*x + y*y);
        res.push_back({x, y, minHeight, maxHeight, dist});
        m[ithBeam] = res.size()-1;
    }  
    return std::make_pair(res, m); 
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

std::vector<std::string> getEvalFilenames(const std::string &root, int frameIdx)
{
    std::vector<std::string> filenames;
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(10) << frameIdx;
    filenames.push_back(root + "/gt_" + ss.str() + "_l.txt"); 
    filenames.push_back(root + "/gt_" + ss.str() + "_r.txt"); 
    return filenames; 
}

cv::viz::WPolyLine generateWPolyLine(std::vector<float> coeffs, float minY, float maxY)
{
    std::vector<cv::Vec3f> linePoints;
    for (int i = minY * 100; i <= maxY * 100; i++) {
        // Check the order of coeffs !
        linePoints.push_back(cv::Vec3f(coeffs[3] + coeffs[2] * i / 100. + coeffs[1] * powf(i / 100., 2) + coeffs[0] * powf(i / 100., 3), i / 100., 0));
    }
    cv::Mat pointsMat = cv::Mat(static_cast<int>(linePoints.size()), 1, CV_32FC3, &linePoints[0]);
    return cv::viz::WPolyLine(pointsMat, cv::viz::Color::blue());
}

cv::viz::WPolyLine generateGTWPolyLine(std::string &filename)
{
    std::ifstream f(filename);
    std::vector<std::vector<float>> data;
    if (f.is_open()) 
    {
        std::string line;
        while (getline(f, line))
        {
            std::stringstream ss(line);
            std::vector<float> v;
            float num;   
            while (ss >> num)
            {
                v.push_back(num);   
            }
            data.push_back(v);
        }
        std::vector<float> y;
        for (int i = 1; i < data.size(); i++)
        {
            y.push_back(data[i][1]);   
        }
        auto minmaxY = std::minmax_element(y.begin(), y.end());
        for (auto & i : data[0])
        {
            std::cout << i << "  ";
        }
        std::cout << std::endl;
        return generateWPolyLine(data[0], *minmaxY.first, *minmaxY.second);
    }
    else
    {
        std::cerr << "GT file not found\n";
    }
    std::vector<cv::Vec3f> zero;
    zero.push_back(cv::Vec3f(0, 0, 0));
    cv::Mat pointsMat = cv::Mat(static_cast<int>(zero.size()), 1, CV_32FC3, &zero[0]);
    return cv::viz::WPolyLine(pointsMat, cv::viz::Color::blue());
}

std::vector<cv::viz::WPolyLine> generateGTWPolyLines(const std::string &root, int frameIdx)
{
    std::vector<cv::viz::WPolyLine> polyLines;
    std::vector<std::string> filenames = getEvalFilenames(root, frameIdx);
    for (std::string &filename : filenames)
    {
        polyLines.push_back(generateGTWPolyLine(filename));
    }
    return polyLines;
}

// std::string evalPath = "/home/rtml/Lidar_curb_detection/source/evaluation/gt_generator/evaluation_result_20191126";
// std::string evalPath = "/home/rtml/Lidar_curb_detection/source/evaluation/gt_generator/evaluation_result_kitti_20110926_0002";
std::string evalPath = "/home/rtml/LiDAR_camera_calibration_work/data/kitti_data/2011_09_26/2011_09_26_drive_0051_sync/0051_gt";

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

    // Boundary detection object : our data
    // int frameStart = 601, frameEnd = 650;
    // Boundary_detection detection(16, 1.125, "/home/rtml/lidar_radar_fusion_curb_detection/data/", "20191126163620_synced_601_650/", frameStart, frameEnd+1, false);
    
    // Boundary detection object : kitti data
    // int frameStart = 30, frameEnd = 50;
    // Boundary_detection detection(64, 1.125, "/home/rtml/LiDAR_camera_calibration_work/data/kitti_data/2011_09_26/", "2011_09_26_drive_0013_sync/", frameStart, frameEnd+1, false);
    
    // Boundary detection object : kitti data
    int frameStart = 0, frameEnd = 50;
    Boundary_detection detection(64, 1.125, "/home/rtml/LiDAR_camera_calibration_work/data/kitti_data/2011_09_26/", "2011_09_26_drive_0051_sync/", frameStart, frameEnd+1, false);
    
    // Virtual scan object
    FastVirtualScan virtualscan = FastVirtualScan();
    fusion::FusionController fuser;

    // Main loop
    int frame_idx = 0;
    while (detection.isRun() && !viewer.wasStopped()) 
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

        // Read in data 
        detection.retrieveData();
        detection.pointcloud_preprocessing(rot_vec[0]); 
        auto &pointcloud = detection.get_pointcloud();
        LidarViewer::pushToBuffer(buffers[0], pointcloud);
        results[0] = std::vector<bool>(pointcloud.size(), false);
        
        auto t_start = std::chrono::system_clock::now();
        // Run virtualscan algorithm 
        QVector<double> beams; 
        virtualscan.calculateVirtualScans(buffers[0], BEAMNUM, STEP, MINFLOOR, MAXCEILING, OBSTACLEMINHEIGHT, MAXBACKDISTANCE, 
                                          ROTATION * PI / 180.0, MINRANGE);

        virtualscan.getVirtualScan(ROADSLOPMINHEIGHT * PI / 180.0, ROADSLOPMAXHEIGHT * PI / 180.0, MAXFLOOR, MINCEILING, 
                                   PASSHEIGHT, beams);

        auto res = getVscanResult(virtualscan, beams);
        auto t_end = std::chrono::system_clock::now();
        auto buf = detection.runDetection(rot_vec[0], trans_vec[0], res.first, res.second);
        
        std::vector<cv::viz::WPolyLine> thirdOrder = detection.getThirdOrderLines(buf[1]);
        results[0] = detection.get_result_bool();

        std::vector<cv::viz::WPolyLine> gtLines = generateGTWPolyLines(evalPath, frameStart+frame_idx);

        LidarViewer::updateViewerFromBuffers(buffers, results, viewer, res.first, thirdOrder, gtLines);

        std::chrono::duration<double, std::milli> fp_ms = t_end - t_start;
        std::cout << "Frame " << frame_idx++ << ": takes " << fp_ms.count() << " ms for vsan" << std::endl;
        if (frame_idx >= 10)
        {
            total_ms += fp_ms.count();
        }
    }
    viewer.close();
    std::cout << "Average time per frame: " << (total_ms / (frame_idx-10)) << " ms\n";
    return 0;
}

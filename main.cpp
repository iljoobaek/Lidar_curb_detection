#include <csignal>

#include "boundary_detection.h"
#include "viewer.h"
#include "sensor_config.h"
#include <exception> 
#include "fastvirtualscan/fastvirtualscan.h"

// Parameters for virtual scan
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

std::string getEvalFilename(const std::string &root, int frameIdx)
{
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(10) << frameIdx;
    std::string filename = root + "/gt_" + ss.str() + "_r.txt"; 
    std::cout<<filename<<" found "<<std::endl;
    return filename; 
}

cv::viz::WPolyLine generateWPolyLine(std::vector<float> coeffs, float minY, float maxY)
{
    std::vector<cv::Vec3f> linePoints;
    for (int i = minY * 100; i <= maxY * 100; i++) {
        // Check the order of coeffs !
        linePoints.push_back(cv::Vec3f(coeffs[3] + coeffs[2] * i / 100. + coeffs[1] * powf(i / 100., 2) + coeffs[0] * powf(i / 100., 3), i / 100., 0));
    }
    std::cout<<"Points pushed \n";
    cv::Mat pointsMat = cv::Mat(static_cast<int>(linePoints.size()), 1, CV_32FC3, &linePoints[0]);
    return cv::viz::WPolyLine(pointsMat, cv::viz::Color::blue());
}

// std::vector<std::string> xyz = glob_new( rootPath.c_str() + std::string(filename) + std::string("velodyne_points/data/*"));
std::vector<std::string> glob_gt(const std::string& pattern)  {
    std::cout<<"Indise \b";
    glob_t globbuf;
    int err = glob(pattern.c_str(), 0, NULL, &globbuf);
    std::vector<std::string> filenames;
    if(err == 0)
    {
        for (size_t i = 0; i < globbuf.gl_pathc; i++)
        {
            filenames.push_back(globbuf.gl_pathv[i]);
        }
        int start_pos = filenames[0].find("_r");
        std::string start_num= filenames[0].substr(start_pos -10 );
        std::cout<<"Start string "<< start_num;
        // int start_frame = std::stoi(start_num);
        // std::cout<<"Start frame is "<< start_frame;

        int end_pos = filenames[filenames.size()-1].find("_r");
        std::string end_num= filenames[filenames.size()-1].substr(end_pos -10 );
        // int end_frame = std::stoi(end_num);
        // std::cout<<"End frame is "<< end_frame;

        globfree(&globbuf);
        return filenames;
    }
    else{
        filenames.push_back("0");
        return filenames;
    }
}



cv::viz::WPolyLine generateGTWPolyLine(const std::string &root, int frameIdx)
{
    std::string filename = getEvalFilename(root, frameIdx);
    std::cout<<"before \n";
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
        std::cout<< "generating gt line -------------------------------------------\n";
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


///////////////////////////////// mbhat //////////////////////////////////////////////

PyObject* vectorToList_Float(const std::vector<float> &data) {
  PyObject* listObj = PyList_New( data.size() );
	if (!listObj) throw std::logic_error("Unable to allocate memory for Python list");
	for (unsigned int i = 0; i < data.size(); i++) {
		PyObject *num = PyFloat_FromDouble( (double) data[i]);
		if (!num) {
			Py_DECREF(listObj);
			throw std::logic_error("Unable to allocate memory for Python list");
		}
		PyList_SET_ITEM(listObj, i, num);
	}
	return listObj;
}
PyObject* NullListObj(void) {
    std::cout<<"Reached here \n";
    PyObject* listObj = PyList_New( 1 );
    PyObject *num = PyFloat_FromDouble( 0.0 );
    PyList_SET_ITEM(listObj, 0, num);
    return listObj;
}

std::vector<float> listTupleToVector_Float(PyObject* incoming) {
	std::vector<float> data;
	if (PyTuple_Check(incoming)) {
		for(Py_ssize_t i = 0; i < PyTuple_Size(incoming); i++) {
			PyObject *value = PyTuple_GetItem(incoming, i);
			data.push_back( PyFloat_AsDouble(value) );
		}
	} else {
		if (PyList_Check(incoming)) {
			for(Py_ssize_t i = 0; i < PyList_Size(incoming); i++) {
				PyObject *value = PyList_GetItem(incoming, i);
				data.push_back( PyFloat_AsDouble(value) );
			}
		} else {
			throw std::logic_error("Passed PyObject pointer was not a list or tuple!");
		}
	}
	return data;
}

std::string pathJoin(std::string p1, std::string p2) {

   char sep = '/';
  if (p1[p1.length()] != sep) { // Need to add a
     p1.push_back(sep);                // path separator
     if(p2[p2.length()] != sep){
        p2.push_back(sep);
     }
  }
  else{
       if(p2[p2.length()] != sep){
        p2.push_back(sep);
    }

  }
    return(p1+ p2);
}

///////////////////////////////////////// End: mbhat ////////////////////////////////////////////////


// Default
// std::string evalPath = "/home/droid/manoj_work/Curb_detect_merge/source/evaluation/accuracy_calculation/evaluation_result_20191126";
std::string evalPath = "/home/droid/manoj_work/Curb_detect_merge/source/evaluation/accuracy_calculation/2011_09_26/2011_09_26_drive_0051_sync";

int main(int argc, char* argv[]) 
{
    // boundary_detection "root_folder" "kitti_date_folder" "kitti_video_folder" "5"
    // Number of velodyne sensors, maximum 6
    
    std::cout<<"STARTED ...\n";
    
    std::string rootGivenPath = "/home/droid/manoj_work/Lidar_curb_detection_full/source";
    std::string kittiDateFolder = "2011_09_26";
    std::string kittiVideoFolder = "2011_09_26_drive_0013_sync";
    int numOfVelodynes;
    if (argc < 2){
        numOfVelodynes = 6;
    }
    if (argc < 5)
    {
        rootGivenPath = argv[1]; // "/home/droid/manoj_work/Lidar_curb_detection_full/source"
        kittiDateFolder = argv[2]; // "2011_09_26"
        kittiVideoFolder = argv[3]; // "2011_09_26_drive_0013_sync"
        numOfVelodynes = 6;
    } 
    else if (argc == 5) 
    {
        numOfVelodynes = std::stoi(argv[2]);
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

    std::string evalPath = pathJoin(pathJoin(pathJoin(rootGivenPath, "evaluation/accuracy_calculation/"),  kittiDateFolder), kittiVideoFolder);
    std::string dataPath = "/home/droid/manoj_work/Lidar_curb_detection_full/kitti_data/";
    std::string datePath = pathJoin(dataPath, kittiDateFolder);
    std::cout<<"Eval path is "<<evalPath<<"\n";
    std::cout<<"Data path is "<<dataPath<<"\n";
    std::cout<<"Date path is "<<datePath<<"\n";
    std::cout<<"Total path is "<<pathJoin(datePath, kittiVideoFolder)<<"\n";
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
    // Boundary_detection detection(16, 1.125, "/home/rtml/lidar_radar_fusion_curb_detection/data/", "20191126163620_synced/", frameStart, frameEnd+1, false);
    
    // Boundary detection object : kitti data How many frames?
    int frameStart = 0, frameEnd = 100; // DOesnt matter, using glob inside datareader
    Boundary_detection detection(64, 1.125, datePath, kittiVideoFolder+"/", frameStart, frameEnd+1, true);
    
    // Virtual scan object
    FastVirtualScan virtualscan = FastVirtualScan();
    fusion::FusionController fuser;

    Py_Initialize();
    PyObject *pName, *pModule, *pDict, *pFunc, *pArgs, *pValue;
    
    // Import tracker
    pModule = PyImport_Import(PyString_FromString("tracking"));
    pDict = PyModule_GetDict(pModule);
    // get functions
    pFunc = PyDict_GetItemString(pDict, "kalman_filter_chk");

    // Location for storing prev lines
    std::vector<float> prev_lcoeff;
    std::vector<float> prev_rcoeff;
    std::vector<float> prev_avail_lcoeff;
    std::vector<float> prev_avail_rcoeff;

    std::vector<cv::Vec3f> zero;
    zero.push_back(cv::Vec3f(0, 0, 1));
    cv::Mat pointsMat = cv::Mat(static_cast<int>(zero.size()), 1, CV_32FC3, &zero[0]);
    cv::viz::WPolyLine prev_lline = cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());
    cv::viz::WPolyLine prev_rline = cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());

    // std::vector<std::string> xyz = glob_gt( root.c_str() + std::string("/*"));

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
        
        std::cout<<"start \n";

        std::vector<std::vector<cv::Vec3f>> buffers(numOfVelodynes); 

        // Read in data 
        std::vector<int> globFrameNums = detection.retrieveData();

        if (globFrameNums[0] != frameStart){
            std::cout<<"frame start and data start missmatch \n";
        }
        std::cout<<"Frame ids : "<<globFrameNums[0]<< " "<< globFrameNums[1]<<"\n";
        frameStart = globFrameNums[0]; // Changing frame start here

        detection.pointcloud_preprocessing(rot_vec[0]); 
        auto &pointcloud = detection.get_pointcloud();
        LidarViewer::pushToBuffer(buffers[0], pointcloud);
        
        auto t_start = std::chrono::system_clock::now();
        // Run virtualscan algorithm
        // Vizualize
        LidarViewer::updateViewerFromBuffersOnlyLidar(buffers, viewer);
        auto t_end = std::chrono::system_clock::now();
        
        std::chrono::duration<double, std::milli> fp_ms = t_end - t_start;
        std::cout << "Frame " << frame_idx++ << ": takes " << fp_ms.count() << " ms for vsan" << std::endl;
        if (frame_idx >= 10)
        {
            total_ms += fp_ms.count();
        }
    }

    // Stop python script
    Py_Finalize();

    viewer.close();
    std::cout << "Average time per frame: " << (total_ms / (frame_idx-10)) << " ms\n";
    return 0;
}

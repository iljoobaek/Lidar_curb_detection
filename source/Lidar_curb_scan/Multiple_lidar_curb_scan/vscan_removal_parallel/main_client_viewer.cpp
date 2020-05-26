#include <csignal>        
#include "boundary_detection.h"
#include "viewer.h"
#include "sensor_config.h"
#include <exception> 
#include "fastvirtualscan/fastvirtualscan.h"
#include "csv.h"
#include <queue> 
#include "argh.h"

#include <sys/time.h>
#include <sys/syscall.h>
#include <sched.h>
#include <pthread.h>

#include <signal.h> 

// #include <boost/interprocess/managed_shared_memory.hpp>
// #include <boost/interprocess/containers/vector.hpp>
// #include <boost/interprocess/allocators/allocator.hpp>
// #include <boost/interprocess/sync/scoped_lock.hpp>
// #include <boost/interprocess/sync/named_mutex.hpp>
// #include <boost/interprocess/sync/named_condition.hpp>
// #include <boost/interprocess/detail/os_thread_functions.hpp>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include "boost/date_time/posix_time/posix_time.hpp" 
#include <boost/thread/thread_time.hpp>



namespace bi = boost::interprocess;

#define GLOBAL_INIT_TIME_DEBUG

#ifdef GLOBAL_INIT_TIME_DEBUG
	#define Global_Init_time_DPNT(fmt, args...)		fprintf(stdout, fmt, ## args)
	#define Global_Init_time_EPNT(fmt, args...)		fprintf(stderr, fmt, ## args)
#else
	#define Global_Init_time_DPNT(fmt, args...)
	#define Global_Init_time_EPNT(fmt, args...)		fprintf(stderr, fmt, ## args)
#endif

int STOP_EXECUTION = 0;
void ctrlhandler(int){STOP_EXECUTION = 1;}
void killthread(int){STOP_EXECUTION = 2;}

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

// Global switches 
int ULTRASONIC = 0;
int DEBUG = 0;

static volatile sig_atomic_t sig_caught = 0;
std::queue<long long> timeStampQueue;
std::vector<long long> timeStampVector;

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
//         res.push_back({x, y, minHeight, maxHeight});
        float dist = std::sqrt(x*x + y*y);
        res.push_back({x, y, minHeight, maxHeight, dist});
        m[ithBeam] = res.size()-1;
    }  
//     return res; 
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

std::vector<std::string> getEvalFilenames(const std::string &root, int frameIdx)
{
    std::vector<std::string> filenames;
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(10) << frameIdx;
    filenames.push_back(root + "/gt_" + ss.str() + "_l.txt"); 
    filenames.push_back(root + "/gt_" + ss.str() + "_r.txt"); 
    return filenames; 
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
    // kittfolder/mainfoler/   datafolder
  char sep = '/';
  if (p1[p1.length()-1] != sep) { // Need to add a
     p1.push_back(sep);                // path separator
     if(p2[p2.length()-1] != sep){
        p2.push_back(sep);
     }
  }
  else{
       if(p2[p2.length()-1] != sep){
        p2.push_back(sep);
    }

  }
    return(p1+ p2);
}

///////////////////////////////////////// End: mbhat ////////////////////////////////////////////////


/* Ultrasonic Reading - Tarang */ // Method to compare which one is the more close.
int getClosestIndex(std::vector<long long> arr, int i1, int i2, int target)
{
    if (target - arr[i1] >= arr[i2] - target)
        return i2;
    else
        return i1;
}

// Returns element closest to target in arr[]
int findClosestIndex(std::vector<long long> arr, long long target)
{
    int n = arr.size();
    // Corner cases
    if (target <= arr[0])
        return 0; //arr[0];
    if (target >= arr[n - 1])
        return n - 1;
    //arr[n - 1];

    // Doing binary search
    int i = 0, j = n, mid = 0;
    while (i < j)
    {
        mid = (i + j) / 2;

        if (arr[mid] == target)
            return mid;
        //val= arr[mid];

        /* If target is less than array element, 
            then search in left */
        if (target < arr[mid])
        {

            // If target is greater than previous
            // to mid, return closest of two
            if (mid > 0 && target > arr[mid - 1])
                return getClosestIndex(arr, mid - 1, mid, target);

            /* Repeat for left half */
            j = mid;
        }
        // If target is greater than mid
        else
        {
            if (mid < n - 1 && target < arr[mid + 1])
                return getClosestIndex(arr, mid,
                                       mid + 1, target);
            // update i
            i = mid + 1;
        }
    }

    // Only single element left after search
    return mid;
}

const std::vector<std::vector<float>> getUltrasonicData(std::vector<std::vector<std::vector<float>>> result,
                                                        std::vector<long long> timestamps, long long timestamp)
{
    //search current ttimestamp in array
    int idx = findClosestIndex(timestamps, timestamp);
    return result[idx];
}

std::vector<cv::Vec3f> cvUltrasonicPoints = {cv::Vec3f(0, 0, 0)};
std::vector<std::vector<std::vector<float>>> ultrasonicPoints;
std::vector<long long> ultrasonicTimestamps;

const void loadUltrasonicPoints(std::string path)
{
    path = path.substr(0, path.length()-1);
    // Read file, return a list of timestamp,x,y,z
    std::vector<std::vector<std::vector<float>>> result;
    std::vector<long long> timestamps;
    io::CSVReader<9> in(path+".csv");
    in.read_header(io::ignore_extra_column, "s1", "s2", "s3", "s4", "fs1", "fs2", "fs3", "fs4", "timestamp");
    long long timestamp;
    float s1, s2, s3, s4, fs1, fs2, fs3, fs4;
    double speed;
    while (in.read_row(s1, s2, s3, s4, fs1, fs2, fs3, fs4, timestamp))
    {
        // do stuff with the data
        // std::cout << timestamp << "," << s1 << "," << s2 << "," << s3 << "," << s4;
        std::vector<std::vector<float>> cur_result;
        float z[4] = {-1.125,-1.125,-1.125,-1.125};
        float y_offset= 2.09;
        float y[4] = {0.66-y_offset, 0.22-y_offset, -0.22-y_offset, -0.66-y_offset};
        float x_offset = 0.91;
        float x[4] = {x_offset+(s1 / 100),
                      x_offset+(s2 / 100),
                      x_offset+(s3 / 100),
                      x_offset+(s4 / 100)};
        float fx[4] = {x_offset+(fs1 / 100),
                       x_offset+(fs2 / 100),
                       x_offset+(fs3 / 100),
                       x_offset+(fs4 / 100)};
        for (int i = 0; i < 4; i++)
        {
            std::vector<float> data{fx[i], y[i], z[i]};
            cur_result.push_back(data);
        }
        result.push_back(cur_result);
        timestamps.push_back(timestamp);
    }
    ultrasonicPoints = result;
    ultrasonicTimestamps = timestamps;
}




void readTimestamps(std::string folderPath)
{
    //binPath = XXX/velodyne_points/

    std::string line;
    std::string path = pathJoin(folderPath, "timestamps.txt");
    path = path.substr(0, path.length()-1);
    std::ifstream myfile(path);
    std::cout<<path<<"\n";
    std::getline(myfile, line);
    for (long long result=std::stoll(line); std::getline(myfile, line); result = std::stoll(line))
    {
        long long newResult = result/1000; // (ms to sec)
        // timeStampQueue.push(newResult);
        timeStampVector.push_back(newResult);
        // std::cout<<timeStampQueue.front()<<"\n";

    }
    std::cout << "Pushed timestamps to queue successfully Size:" << timeStampVector.size() << "\n\n";
}

std::vector<cv::viz::WPolyLine> getCoeffsForTracking(Boundary_detection *detection, std::vector<cv::Vec3f> *buffer, std::vector<float> *curv, int choice){
    if(ULTRASONIC == 1){
        std::vector<cv::Vec3f>* ptr = &cvUltrasonicPoints;
        return detection->getLineFromCoeffs(buffer, ptr, curv, choice); 
    }else{
        return detection->getLineFromCoeffs(buffer, curv, choice); 
    }
    
}


template<typename T> void print_queue_ptr(T &q){
    while(!q->empty()){
       
        q->pop();
    }
    std::cout<<"\n";
}

void capture_and_detect_bool(velodyne::VLP16Capture *capture, std::vector<std::vector<float>> &pointcloud, std::vector<cv::Vec3f> &buffer, std::vector<bool> &result, float theta, cv::Mat &rot, cv::Mat &trans) { 
    std::vector<velodyne::Laser> laser;
    
    // Capture one frame
    (*capture) >> laser;
    // Convert pointcloud to cartesian and copy to detection object 
    LidarViewer::laser_to_cartesian(laser, pointcloud, theta, rot, trans);
    std::cout<<"[LV] size of pcd "<< pointcloud.size()<<"\n";
    // Push result to buffer 
    LidarViewer::push_result_to_buffer_vector(buffer, pointcloud, rot, trans);
    std::cout<<"[LV] size of buffer "<< buffer.size()<<"\n";
    result = std::vector<bool>(pointcloud.size(), false);
}

// Default
// std::string evalPath = "/home/droid/manoj_work/Curb_detect_merge/source/evaluation/accuracy_calculation/evaluation_result_20191126";
std::string evalPath = "/home/droid/manoj_work/Curb_detect_merge/source/evaluation/accuracy_calculation/2011_09_26/2011_09_26_drive_0051_sync";

int main(int argc, char* argv[]) 
{
    // boundary_detection "root_folder" "kitti_date_folder" "kitti_video_folder" "5"
    // Number of velodyne sensors, maximum 6
    
    /* Killing signals */
    signal(SIGINT, ctrlhandler);
    signal(SIGTERM, killthread);


    std::string rootGivenPath ;
    std::string dataPath;
    std::string kittiDateFolder;
    std::string kittiVideoFolder;

    // rootGivenPath = "/home/droid/manoj_work/Lidar_curb_detection_full/source";
    // dataPath =  "/home/droid/manoj_work/Lidar_curb_detection_full/kitti_data/";
    // kittiDateFolder = "2011_09_26";
    // kittiVideoFolder = "2011_09_26_drive_0013_sync";
    rootGivenPath = "/home/droid/manoj_work/Lidar_curb_detection_full/source";
    dataPath =  "/home/droid/manoj_work/Lidar_curb_detection_full/kitti_data/";
    kittiDateFolder = "2011_09_26";
    kittiVideoFolder = "2011_09_26_drive_0013_sync";
    int lidar_num = 0;
    float rot_deg = 0;
    std::string direction = "center_front";
    int PCAP = 0;
     

    
    std::string keys =
    "{rootPath |/path_to_src   | input image path}"         
    "\n{data  |/data_folder_name | face cascade path}"       
    "\n{date   | /date_folder_name | eye cascade path}"
    "\n{video   | /video_folder_name | eye cascade path}";
    "\n{lidar   | /video_folder_name | eye cascade path}";

    argh::parser commandline = argh::parser(argc, argv, argh::parser::PREFER_PARAM_FOR_UNREG_OPTION);
    commandline.add_param({"-root, --rootPath"});
    commandline.add_param({"-data, --dataName"});
    commandline.add_param({"-date, --dateFolder"});
    commandline.add_param({"-video, --videoFolder"});
    commandline.add_param({"-lidar, --lidarNum"});
    commandline.add_param({"-dir, --direction"});
    commandline.add_param({"-rot, --rotation"});
    commandline.add_param({"-pcap, --pcapfile"});
    commandline.parse(argc, argv);    

    if (commandline["v"]){
        DEBUG=1;
        std::cout<<"*** DEBUG *** \n";
    }
    if (commandline["u"]){
        ULTRASONIC=1;
        std::cout<<"*** ULTRASONIC *** \n";
    }
    if(commandline("root")){
        rootGivenPath = commandline("root").str();
    }
    if(commandline("data")){
        dataPath = commandline("data").str();
    }
    if(commandline("date")){
        kittiDateFolder = commandline("date").str();
    }
    if(commandline("video")){
        kittiVideoFolder = commandline("video").str();
    }
    if(commandline("lidar")){
        lidar_num = std::stoi(commandline("lidar").str());
    }
    if(commandline("dir")){
        direction = commandline("dir").str();
    }
    if(commandline("rot")){
        rot_deg = std::stof(commandline("rot").str());
    }
    if(commandline("pcap")){
        PCAP = 1;
    }


    std::string pcap_file;
    float rot_param;
    cv::Mat rot_vec;
    cv::Mat trans_vec;
    std::vector<float> rot_params = SensorConfig::getRotationParams();
    std::vector<cv::Mat> rot_vecs = SensorConfig::getRotationMatrices(rot_params);
    std::vector<cv::Mat> trans_vecs = SensorConfig::getTranslationMatrices();
    velodyne::VLP16Capture* capture;
    if(PCAP == 1){
        // Get pcap file names
        std::vector<std::string> pcap_files = SensorConfig::getPcapFiles(); 
        // Get rotation and translation parameters from lidar to vehicle coordinate
        
        std::cout<<"Debugk\n";
        std::cout<<"Lidar num "<<lidar_num<<"\n";
        switch((int)lidar_num){
            case 0:
                direction = "center_front";
                break;
            case 1:
                direction = "center_rear";
                break;
            case 2:
                direction = "left_front";
                break;
            case 3:
                direction = "left_side";
                break;
            case 4:
                direction = "right_front";
                break;
            case 5:
                direction = "right_side";
                break;
            default:
                std::cout<<"Wrong direction, try again;\n";
                return 0;
        }
        pcap_file = pcap_files[lidar_num];
        rot_param = rot_params[lidar_num];
        rot_vec = rot_vecs[lidar_num];
        trans_vec = trans_vecs[lidar_num];
        std::cout<<"[DISPLAY] PCAP file path :"<<pcap_file.c_str()<<"\n";

        // Read pcap file 
        capture = new velodyne::VLP16Capture(pcap_file); 
    }

    // Make necessary paths
    std::string evalPath = pathJoin(pathJoin(pathJoin(rootGivenPath, "evaluation"),  kittiDateFolder), kittiVideoFolder);
    std::string datePath = pathJoin(dataPath, kittiDateFolder);
    
    // Display paths
    std::cout<<"Eval path is "<<evalPath<<"\n";
    std::cout<<"Data path is "<<dataPath<<"\n";
    std::cout<<"Date path is "<<datePath<<"\n";
    std::cout<<"Total path is "<<pathJoin(datePath, kittiVideoFolder)<<"\n";

    // Signal Handler for pause/resume viewer
    std::signal(SIGINT, signalHandler);
    double total_ms = 0.0;
    int frameStart = 10, frameEnd = 200; // Doesnt matter, using glob inside datareader
    long long timeStamp;

    // Create Viz3d Viewer and register callbacks
    cv::viz::Viz3d viewer( "Velodyne front" );
    bool pause(false);
    LidarViewer::cvViz3dCallbackSetting(viewer, pause);
    std::cout<<"Debug5\n";
    

    std::cout<<"[DEBUG] Making boundary detection object \n";
    // ::Boundary detect
    // for kitti 16 data
    // Boundary_detection detection(64, 1.125, datePath, kittiVideoFolder+"/", frameStart, frameEnd+1, false);
    Boundary_detection detection(16, 1.125, pcap_file);
    // for our data
    // Boundary_detection detection(16, 1.125, datePath, kittiVideoFolder+"/", frameStart, frameEnd+1, false);
    std::cout<<"[DEBUG] Virtual scan object \n";
    // Virtual scan object
    FastVirtualScan virtualscan = FastVirtualScan();
    fusion::FusionController fuser;
    
    // Main Event loop
    int frame_idx = 0;
    std::string mem_name;
    std::vector<std::vector<cv::Vec3f>> pointcloud_buffer(1);
    bool noTimeout = false;
    boost::system_time timeout;
        std::cout<<"[DEBUG] Main loop start \n";
            std::vector<cv::viz::WPolyLine> thirdOrder;
    
    // bi::MutexType mutex;
    while(!viewer.wasStopped() && STOP_EXECUTION == 0)
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
        
        pointcloud_buffer.clear();
        pointcloud_buffer.resize(1);

        std::cout<<"Waiting now \n";
        
        std::cout<<"Direction:  "<< direction<<"\n";
        
        std::vector<cv::Vec3f> buffer;
        std::vector<bool> result;
        std::pair<std::vector<std::vector<float>>, std::unordered_map<int, int>> vcan_results;
        std::vector<std::vector<cv::Vec3f>> boundary_buffer;
        std::vector<std::vector<float>> pointcloud_copy;
        std::vector<std::vector<bool>> results_front(1);
        if(PCAP){

            std::cout<<"[DEBUG] Capturing from pcap \n";
            capture_and_detect_bool(capture, pointcloud_copy, buffer, result, rot_param, rot_vec, trans_vec);
            

            // std::cout<<"[DEBUG] Pushing to detection \n";
            // detection.put_pointcloud(pointcloud_copy);
            // detection.pointcloud_preprocessing_pcap();
            // std::cout<<"[DEBUG] pcd det size is "<< pointcloud_copy.size() <<" \n";
            
            std::cout<<"[DEBUG] Getting from detection \n";
            // detection.put_pointcloud(buffer);
            // detection.put_pointcloud(pointcloud_copy);
            // std::vector<int> globFrameNums = detection.retrieveData("front");
            detection.retrieveData();
            // detection.pointcloud_preprocessing(rot_vec); 
            detection.pointcloud_preprocessing_pcap();
            pointcloud_copy = detection.get_pointcloud();
            std::cout<<"[DEBUG] pcd det size is "<< pointcloud_copy.size() <<" \n";
            LidarViewer::pushToBuffer(buffer, pointcloud_copy);
            // detection.pointcloud_preprocessing_pcap();



            std::cout<<"[DEBUG] Vscan loop \n";
            QVector<double> beams; 
            virtualscan.calculateVirtualScans(buffer, BEAMNUM, STEP, MINFLOOR, MAXCEILING, OBSTACLEMINHEIGHT, MAXBACKDISTANCE, 
                                            ROTATION * PI / 180.0, MINRANGE);
            virtualscan.getVirtualScan(ROADSLOPMINHEIGHT * PI / 180.0, ROADSLOPMAXHEIGHT * PI / 180.0, MAXFLOOR, MINCEILING, 
                                    PASSHEIGHT, beams);
            vcan_results = getVscanResult(virtualscan, beams);
            std::cout<<"[DEBUG] Getting boundary buffers \n";
            // boundary_buffer = detection.getBoundaryBuffersSingle( vcan_results.first, vcan_results.second, kittiVideoFolder);
            boundary_buffer = detection.getBoundaryBuffers(rot_vec, trans_vec, vcan_results.first, vcan_results.second, kittiVideoFolder);
            // pointcloud_copy = detection.get_pointcloud();
            LidarViewer::pushToBuffer(buffer, pointcloud_copy);
            std::cout<<"[DEBUG] Bounday buffer size is "<< boundary_buffer[1].size() <<" \n";
            std::cout<<"[DEBUG] pcd size is "<< pointcloud_copy.size() <<" \n";
            
            
            std::cout<<"[DEBUG] size \n";
            results_front[0] = std::vector<bool>(pointcloud_copy.size(), false);
            std::cout<<"[DEBUG] result bool \n";
            results_front[0] = detection.get_result_bool();
            std::cout<<"[DEBUG] size "<<results_front[0].size() <<" \n";
            thirdOrder = detection.getThirdOrderLines(boundary_buffer[1]);

            // std::vector<std::vector<bool>> results_front(2);
            // results_front[0] = std::vector<bool>(pointcloud_copy.size(), false);
            // results_front[0] = detection.get_result_bool(); 

            // // void pointcloud = put_pointcloud(pointcloud_buffer[k]);
            // thirdOrder_lines.reserve(2);
            // fuser.displayThirdOrderAndBuffer(boundary_buffer, boundary_display_buffer, thirdOrder_lines[k]);

        
        
        }else{
            std::vector<int> globFrameNums = detection.retrieveData(direction);
            detection.pointcloud_preprocessing(rot_vecs[0]); 
            std::vector<std::vector<float>> &pointcloud = detection.get_pointcloud();
            pointcloud_copy.resize(pointcloud.size());
            std::copy(pointcloud_copy.begin(), pointcloud_copy.end(), std::back_inserter(pointcloud));

            if(direction == "back"){
                auto flip_rot = SensorConfig::getRotationMatrixFromTheta(180.0);
                detection.rotate_and_translate_multi_lidar_yaw(flip_rot);
            }

            std::vector<std::vector<float>> &pointcloud_temp = detection.get_pointcloud();
            LidarViewer::pushToBuffer(buffer, pointcloud_temp);
            QVector<double> beams; 
            virtualscan.calculateVirtualScans(buffer, BEAMNUM, STEP, MINFLOOR, MAXCEILING, OBSTACLEMINHEIGHT, MAXBACKDISTANCE, 
                                            ROTATION * PI / 180.0, MINRANGE);
            virtualscan.getVirtualScan(ROADSLOPMINHEIGHT * PI / 180.0, ROADSLOPMAXHEIGHT * PI / 180.0, MAXFLOOR, MINCEILING, 
                                    PASSHEIGHT, beams);
            vcan_results = getVscanResult(virtualscan, beams);
            boundary_buffer = detection.getBoundaryBuffersSingle( vcan_results.first, vcan_results.second, kittiVideoFolder);
            std::vector<std::vector<bool>> results_front(2);
            results_front[0] = std::vector<bool>(pointcloud_temp.size(), false);
            results_front[0] = detection.get_result_bool(); 
        }

        // std::cout<<"[DEBUG] Putting into buffer \n";
        // int i=0;
        // pointcloud_buffer[0].resize((int)pointcloud_copy.size());
        // for(auto it = pointcloud_copy[0].begin(); it!=pointcloud_copy[0].end(); it = it+3){
        //     pointcloud_buffer[0][i] = cv::Vec3f(*it, *(it+1), *(it+2));
        //     i++; 
        // }

        frame_idx++;
        std::cout << "Updating view ... \n";
         std::cout << "Showing only lidar ... \n";
        std::vector<std::vector<cv::Vec3f>> buffs(1);
        buffs.push_back(buffer);
        // LidarViewer::updateViewerFromBuffersOnlyLidar(buffs, viewer);
        // LidarViewer::updateViewerFromBuffersBoundary(buffs, vcan_results.first, viewer); //boundary_buffer
        // LidarViewer::updateViewerFromBuffersBoundary(buffs, results_front, vcan_results.first, viewer); //boundary_buffer
        LidarViewer::updateViewerFromBuffersTest(buffs, boundary_buffer, vcan_results.first, thirdOrder, viewer); //boundary_buffer

    }
    viewer.close();
    std::cout << "Average time per frame: " << (total_ms / (frame_idx-10)) << " ms\n";
    return 0;
}

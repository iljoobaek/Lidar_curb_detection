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


// #include "global_init_time.h"
namespace bi = boost::interprocess;

//#define GLOBAL_INIT_TIME_DEBUG

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


std::pair<std::vector<std::vector<float>>, std::unordered_map<int, int>> getVscanResults(std::vector<double> &minheights, std::vector<double> &maxheights, std::vector<double> &beams)
{
    std::vector<std::vector<float>> res;
    std::unordered_map<int, int> m;
    double density = 2 * PI / BEAMNUM;
    for (int i = 0; i < BEAMNUM; i++)
    {
        double theta = i * density - PI;
        if (beams[i] == 0 || minheights[i] == maxheights[i])
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
        float minHeight = minheights[i];
        float maxHeight = maxheights[i];
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


// Default
// std::string evalPath = "/home/droid/manoj_work/Curb_detect_merge/source/evaluation/accuracy_calculation/evaluation_result_20191126";
std::string evalPath = "/home/droid/manoj_work/Curb_detect_merge/source/evaluation/accuracy_calculation/2011_09_26/2011_09_26_drive_0051_sync";

int main(int argc, char* argv[]) 
{
    // boundary_detection "root_folder" "kitti_date_folder" "kitti_video_folder" "5"
    // Number of velodyne sensors, maximum 6
    
    std::string rootGivenPath ;
    std::string dataPath;
    std::string kittiDateFolder;
    std::string kittiVideoFolder;

    rootGivenPath = "/home/droid/manoj_work/Lidar_curb_detection_full/source";
    dataPath =  "/home/droid/manoj_work/Lidar_curb_detection_full/kitti_data/";
    kittiDateFolder = "2011_09_26";
    kittiVideoFolder = "2011_09_26_drive_0013_sync";
    int numOfLidars =1 ;
    
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
    commandline.add_param({"-numoflidars, --numOfLidars"});
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
    if(commandline("numoflidars")){
        numOfLidars = std::stoi(commandline("numoflidars").str());
    }

    // Make necessary paths
    std::string evalPath = pathJoin(pathJoin(pathJoin(rootGivenPath, "evaluation"),  kittiDateFolder), kittiVideoFolder);
    std::string datePath = pathJoin(dataPath, kittiDateFolder);
    
    // Display paths
    std::cout<<"Eval path is "<<evalPath<<"\n";
    std::cout<<"Data path is "<<dataPath<<"\n";
    std::cout<<"Date path is "<<datePath<<"\n";
    std::cout<<"Total path is "<<pathJoin(datePath, kittiVideoFolder)<<"\n";

    // try{
    // Signal Handler for pause/resume viewer
    std::signal(SIGINT, signalHandler);
    double total_ms = 0.0;
    std::cout<<"Debug4\n";
    
    // Create Viz3d Viewer and register callbacks
    cv::viz::Viz3d viewer( "Velodyne front" );
    bool pause(false);
    LidarViewer::cvViz3dCallbackSetting(viewer, pause);
    std::cout<<"Debug5\n";
    
    // Get rotation and translation parameters from lidar to vehicle coordinate
    auto rot_params = SensorConfig::getRotationParams();
    auto rot_vec = SensorConfig::getRotationMatrices(rot_params);
    auto trans_vec = SensorConfig::getTranslationMatrices();
    std::cout<<"Debugk\n";
    
    // Boundary detection object : kitti data How many frames?
    readTimestamps(pathJoin(pathJoin(datePath, kittiVideoFolder), "velodyne_points"));
    int frameStart = 10, frameEnd = 200; // Doesnt matter, using glob inside datareader
    long long timeStamp;
    
    // ::Boundary detect
    // for kitti 16 data
    Boundary_detection detection(64, 1.125, datePath, kittiVideoFolder+"/", frameStart, frameEnd+1, false);
    // for our data
    // Boundary_detection detection(16, 1.125, datePath, kittiVideoFolder+"/", frameStart, frameEnd+1, false);
    std::cout<<"Debug6\n";
    // Virtual scan object
    FastVirtualScan virtualscan_front = FastVirtualScan();
    std::cout<<"Vscan 1 done ";
    FastVirtualScan virtualscan_back = FastVirtualScan();
    fusion::FusionController fuser;
    std::cout<<"Vscan 2 done ";

    std::vector<cv::Vec3f> zero;
    zero.push_back(cv::Vec3f(0, 0, 1));
    cv::Mat pointsMat = cv::Mat(static_cast<int>(zero.size()), 1, CV_32FC3, &zero[0]);
    std::vector<float> prev_lcoeff_front;
    std::vector<float> prev_rcoeff_front;
    std::vector<float> prev_avail_lcoeff_front;
    std::vector<float> prev_avail_rcoeff_front;
    cv::viz::WPolyLine prev_lline_front = cv::viz::WPolyLine(pointsMat, cv::viz::Color::black());
    cv::viz::WPolyLine prev_rline_front = cv::viz::WPolyLine(pointsMat, cv::viz::Color::black());
    std::vector<float> prev_lcoeff_back;
    std::vector<float> prev_rcoeff_back;
    std::vector<float> prev_avail_lcoeff_back;
    std::vector<float> prev_avail_rcoeff_back;
    cv::viz::WPolyLine prev_lline_back = cv::viz::WPolyLine(pointsMat, cv::viz::Color::black());
    cv::viz::WPolyLine prev_rline_back = cv::viz::WPolyLine(pointsMat, cv::viz::Color::black()); 

    typedef bi::allocator<float, bi::managed_shared_memory::segment_manager> ShmemAllocator;
    typedef bi::vector<float, ShmemAllocator> bVector;

    std::pair<bVector*, bi::managed_shared_memory::size_type> res;
    bVector *bVec,  *bVec_bndr, *bVec_vscan;
    std::string mem_name;
    std::vector<std::vector<std::vector<float>>> Vscan_results;
    std::vector<std::vector<cv::Vec3f>> pointcloud_buffer(numOfLidars);
    std::vector<std::vector<cv::Vec3f>> pointcloud_bndr_buffer(numOfLidars);
    std::vector<std::vector<cv::viz::WPolyLine>> thirdOrder_lines;
    std::vector<std::vector<std::vector<cv::Vec3f>>> boundary_display_buffer;

    // Main Event loop
    int frame_idx = 0;
    for(int i=0; i<numOfLidars ; i++){
        mem_name = "Pointcloud_shared_memory_"+std::to_string(i);
        bi::shared_memory_object::remove(mem_name.c_str());
        mem_name = "Boundary_shared_memory_"+std::to_string(i);
        bi::shared_memory_object::remove(mem_name.c_str());
        mem_name = "Vscan_shared_memory_"+std::to_string(i);
        bi::shared_memory_object::remove(mem_name.c_str());
        mem_name = "Lidar_update_status_"+std::to_string(i);
        bi::shared_memory_object::remove(mem_name.c_str());
    }
    
    // const int pid = boost::interprocess::detail::get_current_process_id(); 
    bi::interprocess_mutex *mtx_pcd;
    bi::interprocess_mutex *mtx_pcd_bndr;
    bi::interprocess_mutex *mtx_vscan;
    bi::interprocess_condition *cnd_pcd;

    while(!viewer.wasStopped() && STOP_EXECUTION == 0)
    {
        // std::cout << "Here:\n";
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
        pointcloud_bndr_buffer.clear();
        Vscan_results.clear();
        thirdOrder_lines.clear();
        boundary_display_buffer.clear();
        boundary_display_buffer.resize(numOfLidars);
        thirdOrder_lines.resize(numOfLidars);
        pointcloud_buffer.resize(numOfLidars);
        pointcloud_bndr_buffer.resize(numOfLidars);
        Vscan_results.resize(numOfLidars);
        
        int i = 0;
        int j = 0;
        int vscan_size =0;
        for(int k=0; k<numOfLidars && STOP_EXECUTION == 0; k++){
            
            // mem_name = "Lidar_process_mutex_"+std::to_string(k);
            // boost::interprocess::named_mutex nm(bi::open_only, mem_name.c_str());

            // std::cout << "Waiting for lock" << std::endl;
            // nm.lock();
            // std::cout << "Acquired lock" << std::endl;

            std::cout << "Finding segment:\n";
            mem_name = "Pointcloud_shared_memory_"+std::to_string(k);
            bi::managed_shared_memory segment_pcd(bi::open_or_create ,mem_name.c_str(),155000*100);  //segment name

            mem_name = "Boundary_shared_memory_"+std::to_string(k);
            bi::managed_shared_memory segment_bndr(bi::open_or_create ,mem_name.c_str(),155000*100);  //segment name

            mem_name = "Vscan_shared_memory_"+std::to_string(k);
            bi::managed_shared_memory segment_vscan(bi::open_or_create ,mem_name.c_str(),155000*100);  //segment name

            // std::cout << "Creating allocator:\n";
            // mem_name = "Direction_"+std::to_string(k);
            // bi::managed_shared_memory segment_dir(bi::open_or_create ,mem_name.c_str(),1000);
            // std::cout << "Finding status:\n";
            // char  *dir=segment_dir.find<char>("direction").first;
            // std::string direction( dir );

            mem_name = "Pointcloud_shared_mutex_"+std::to_string(k);
            mtx_pcd = segment_pcd.find_or_construct<bi::interprocess_mutex>(mem_name.c_str())();
            std::cout << "Locking pcd\n";
            // mtx_pcd->lock();
            // mtx_pcd->timed_lock(boost::get_system_time() + boost::posix_time::seconds(1));

            mem_name = "Pointcloud_shared_cnd_"+std::to_string(k);
            cnd_pcd = segment_pcd.find_or_construct<bi::interprocess_condition>(mem_name.c_str())();
            std::cout << "Locking pcd\n";
            bi::scoped_lock<bi::interprocess_mutex> lock{*mtx_pcd};

            // mem_name = "Boundary_shared_mutex_"+std::to_string(k);
            // mtx_pcd_bndr = segment_bndr.find_or_construct<bi::interprocess_mutex>(mem_name.c_str())();
            // std::cout << "Locking pcd bndr\n";
            // mtx_pcd_bndr->lock();

            // mem_name = "Vscan_shared_mutex_"+std::to_string(k);
            // mtx_vscan = segment_vscan.find_or_construct<bi::interprocess_mutex>(mem_name.c_str())();
            // std::cout << "Locking vscan\n";
            // mtx_vscan->lock();


            // For the complete pointcloud buffer
            mem_name = "Pointcloud_shared_"+std::to_string(k);
            std::cout<<"Finding for "<<mem_name.c_str()<<"\n";
            res = segment_pcd.find<bVector>( mem_name.c_str() );
            std::cout<<"second : "<<res.second<<"\n";
            if(res.first){
                bVec = res.first;
                std::cout << "Size of bvec :"<<bVec->size()<<"\n";
                if(bVec->size() >0 ){
                    std::cout << "Capacity of bvec :"<<bVec->capacity()<<"\n";
                    i = 0;
                    pointcloud_buffer[k].resize((int)bVec->size());
                    for(auto it = bVec->begin(); it!=bVec->end(); it = it+3){
                        pointcloud_buffer[k][i] = cv::Vec3f(*it, *(it+1), *(it+2));
                        i++; 
                    }

                    std::cout << "Pushed successfully \n";
                    std::cout << "size "<<pointcloud_buffer.size()<<" "<<pointcloud_buffer[k].size()<<"\n";
                }else{
                    std::cout << "EMPTY Pointcloud_shared_ \n";
                }
                
            }else {
                std::cout<<"The result of" + mem_name + "is empty \n";
                std::cout << "Here \n";
                pointcloud_buffer[k].resize(1);
                std::cout << "Here2 \n";
                pointcloud_buffer[k][0] = cv::Vec3f(0,0,0);
            }
            std::cout << "done pcd\n";
            
            // For the boundary buffer
            mem_name = "Boundary_shared_"+std::to_string(k);
            res = segment_bndr.find<bVector>( mem_name.c_str() );
            if(res.first){
                bVec_bndr = res.first;
                i = 0;
                if(bVec_bndr->size() >0 ){
                    pointcloud_bndr_buffer[k].resize((int)bVec_bndr->size()/3);
                    for(auto it = bVec_bndr->begin(); it!=bVec_bndr->end(); it = it+3){
                        pointcloud_bndr_buffer[k][i] = cv::Vec3f((float)*it, (float)*(it+1), (float)*(it+2));
                        i++; 
                    }
                    std::cout << " Data is "<<pointcloud_bndr_buffer[k].size()<<" and "<<pointcloud_bndr_buffer[k][0][0]<<"\n";
                    std::cout << "Pushed successfully \n";
                    
                    // if(direction == "back"){
                    //     // Rotate bndr buffers
                    //     auto flip_rot = SensorConfig::getRotationMatrixFromTheta(180.0);
                    //     detection.rotate_and_translate_multi_lidar_yaw_return(flip_rot, pointcloud_bndr_buffer[k]);
                    // }
                }
                else{
                    std::cout << "EMPTY Boundary_shared_ \n";
                }
                

            }else{
                std::cout<<"The result of" + mem_name + "is empty \n";
                pointcloud_bndr_buffer[k].resize(1);
                pointcloud_bndr_buffer[k][0] = cv::Vec3f(0, 0, 0);
            }
            std::cout << "done pcd bndr\n";
            
            // For the Vscan results
            mem_name = "Vscan_shared_"+std::to_string(k);
            res = segment_vscan.find<bVector>( mem_name.c_str() );
            std::cout << "Searched successfully \n";
            if(res.first){
                bVec_vscan = res.first;
                if(bVec_vscan->size() >0 ){
                    vscan_size += (int)bVec_vscan->size()/4;
                    Vscan_results[k].resize(vscan_size);
                    for(auto it = bVec_vscan->begin(); it!=bVec_vscan->end(); it=it+4){
                        Vscan_results[k][j].resize(4);
                        Vscan_results[k][j][0] = (float)*it;
                        Vscan_results[k][j][1] = (float)*(it+1);
                        Vscan_results[k][j][2] = (float)*(it+2);
                        Vscan_results[k][j][3] = (float)*(it+3);
                        j++;
                    }
                    std::cout << "Pushed successfully \n";
                }else{
                    std::cout << "EMPTY Vscan_shared_ \n";
                }
            }else{
                std::cout<<"The result of" + mem_name + "is empty \n";
                Vscan_results[k].resize(1);
                Vscan_results[k][0].resize(4);
                Vscan_results[k][0][0] = 0;
                Vscan_results[k][0][1] = 0;
                Vscan_results[k][0][2] = 0;
                Vscan_results[k][0][3] = 0;
            }
            std::cout << "done vscan\n";
            // std::cout << "done vscan num : "<<Vscan_results[k][j][0]<<"\n";

            // std::cout << "Unlocking pcd\n";
            // mtx_pcd->unlock();
            // std::cout << "Unlocking pcd bndr\n";
            // mtx_pcd_bndr->unlock();
            // std::cout << "unlocking vscan\n";
            // mtx_vscan->unlock();
            std::cout << "Notify\n";
            cnd_pcd->notify_all();
            // cnd_pcd->wait(lock); 

            std::cout << "Finished making stuff... \n";
            // void pointcloud = put_pointcloud(pointcloud_buffer[k])
            // thirdOrder_lines[k] = detection.getThirdOrderLines(pointcloud_bndr_buffer[k]); // Vizualizing [left, right]

            // if(res.first && bVec_bndr->size() >0){
            //     thirdOrder_lines[k].reserve(2);
            //     fuser.displayThirdOrderAndBuffer(pointcloud_bndr_buffer[k], boundary_display_buffer[k], thirdOrder_lines[k]);
            //     std::cout << "Getting line done soon \n";
            // }
            // nm.unlock();
            thirdOrder_lines[k] = fuser.displayThirdOrder(pointcloud_bndr_buffer[k]);
        }
        
        frame_idx++;
        std::cout << "Updating view ... \n";
        
        // if(res.first && bVec_bndr->size() >0){
        //     LidarViewer::updateViewerFromBuffersManager(pointcloud_buffer, viewer, Vscan_results, thirdOrder_lines[0], thirdOrder_lines[1], boundary_display_buffer[0][0], boundary_display_buffer[0][1], boundary_display_buffer[1][0],boundary_display_buffer[1][1]);
        // }else{
            std::cout << "Showing all lidar ... \n";
            // LidarViewer::updateViewerFromBuffersOnlyLidar(pointcloud_buffer, viewer);
            LidarViewer::updateViewerFromBuffersManagerAll(pointcloud_buffer, pointcloud_bndr_buffer, Vscan_results, thirdOrder_lines, viewer);
        // }
        
    }

    // }
    // catch(bi::interprocess_exception &ex){
    //     std::cout<<ex.get_error_code() <<" error \n";
    //     if(ex.get_error_code() == bi::not_found_error){
    //         std::cout<<"Not found, Continuing... \n";
    //         continue;
    //     }
    //     else{
    //         throw;
    //     }
    // }

    viewer.close();
    std::cout << "Average time per frame: " << (total_ms / (frame_idx-10)) << " ms\n";
    return 0;
}



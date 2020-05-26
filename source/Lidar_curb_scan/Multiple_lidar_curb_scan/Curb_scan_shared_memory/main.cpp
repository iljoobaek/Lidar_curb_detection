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

namespace bi = boost::interprocess;

#define GLOBAL_INIT_TIME_DEBUG

#ifdef GLOBAL_INIT_TIME_DEBUG
	#define Global_Init_time_DPNT(fmt, args...)		fprintf(stdout, fmt, ## args)
	#define Global_Init_time_EPNT(fmt, args...)		fprintf(stderr, fmt, ## args)
#else
	#define Global_Init_time_DPNT(fmt, args...)
	#define Global_Init_time_EPNT(fmt, args...)		fprintf(stderr, fmt, ## args)
#endif



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


void trackerInit(int &frame_num, Boundary_detection &detection, std::vector<std::vector<cv::Vec3f>> &linePoints, PyObject *pFunc, PyObject *pArgs, PyObject *pValue,
            std::vector<float> &prev_avail_lcoeff, cv::viz::WPolyLine &prev_lline, std::vector<float> &prev_lcoeff, std::vector<float> &prev_avail_rcoeff, 
            cv::viz::WPolyLine &prev_rline, std::vector<float> &prev_rcoeff, std::vector<cv::viz::WPolyLine> &thirdOrder, std::vector<std::vector<cv::Vec3f>> &buf,
            std::vector<int> &counts){
        
        std::vector<float> lcoeff = detection.getLeftBoundaryCoeffs(); 
        std::vector<float> rcoeff = detection.getRightBoundaryCoeffs();
        prev_lcoeff = lcoeff;
        prev_rcoeff = rcoeff;
        std::vector<float> v_full;
        std::vector<float> v_left;
        std::vector<float> v_right;
        int leftPointCount = counts[0];
        int rightPointCount = counts[1];

        for(size_t i=0; i<linePoints[0].size(); ++i) 
        {
            const cv::Vec3f& c = linePoints[0][i];
            v_left.push_back(c[0]);
            v_left.push_back(c[1]);
            v_left.push_back(c[2]);   
            std::cout<< v_left.size() << "\t";
        }
        for(size_t i=0; i<linePoints[1].size(); ++i) 
        {
            const cv::Vec3f& c = linePoints[1][i];
            v_right.push_back(c[0]);
            v_right.push_back(c[1]);
            v_right.push_back(c[2]);   
            std::cout<< v_right.size() << "\t";
        }
        
        // Debug counts 
        // if(v_right.size() < 50){
        //         emptyRPointsCount++;
        //     }
        // if(v_left.size() < 50){
        //         emptyLPointsCount++;
        //     }

        if ( lcoeff.empty() || rcoeff.empty() ) {
            std::cout<< "EMPTY ...." << "\n";
            //emptyCount++; 
            
            /// LEFT
            if(lcoeff.empty()){
                std::cout<< "EMPTY .... LEFT" << "\n";
                if (prev_avail_lcoeff.empty()){
                    printf("Lcoeff is empty \n");
                    //emptyLCount++;
                    std::vector<cv::Vec3f> zero;
                    zero.push_back(cv::Vec3f(0, 0, 1));
                    cv::Mat pointsMat = cv::Mat(static_cast<int>(zero.size()), 1, CV_32FC3, &zero[0]);
                    cv::viz::WPolyLine justLine = cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());
                    thirdOrder.push_back(justLine); //left
                    
                }
                else{
                    
                    std::cout<< "EMPTY .... LEFT PREV" << "\n";
                    //emptyLPrevCount++;
                    std::cout<<"LEFT SIZE: "<< prev_avail_lcoeff.size()<< "\n";
                    prev_lcoeff = prev_avail_lcoeff;
                    std::cout<< "EMPTY LEFT HERE " << prev_avail_lcoeff[0]<< "\t"<< prev_avail_lcoeff[1] <<"\n";
                    std::vector<cv::viz::WPolyLine> resLine = detection.getLineFromCoeffs(buf[1], prev_lcoeff, 0);
                    if(detection.getProblem()){
                        std::cout<<"\nPROBLEM 1 \n";
                        thirdOrder.push_back(prev_lline);
                    }else{
                        thirdOrder.push_back(resLine[0]);
                        prev_lline = resLine[0];
                    }
                }
            }
            else{
                // Update
                /////////////////////////////////////////
                if(leftPointCount > 50){
                    std::cout<<"Left update \n";
                    //emptyLUpCount++;
                    pArgs = PyTuple_New(5);
                    PyObject* pointsLeft = vectorToList_Float(v_left);
                    PyObject* pointsRight = vectorToList_Float(v_right);
                    PyObject* lList = vectorToList_Float(lcoeff);
                    PyObject* rList = NullListObj();
                    pValue = PyFloat_FromDouble(frame_num);

                    PyTuple_SetItem(pArgs, 0, pointsLeft);
                    PyTuple_SetItem(pArgs, 1, pointsRight);
                    PyTuple_SetItem(pArgs, 2, lList);
                    PyTuple_SetItem(pArgs, 3, rList);
                    PyTuple_SetItem(pArgs, 4, pValue);

                    int update_flag = 0;

                    try {
                        if(PyCallable_Check(pFunc)){
                            // Update line 
                            std::cout<<"Able to call \n";
                            PyObject * pResult = PyObject_CallObject(pFunc, pArgs);
                            // updateNum++;
                            std::vector<float> res_curve = listTupleToVector_Float(pResult);
                            std::cout<<"Got points Lorig "<<lcoeff[0]<<"\t"<<lcoeff[1]<<"\t"<<lcoeff[2]<<"\n";
                            std::cout<<"Got points "<<res_curve[0]<<"\t"<<res_curve[1]<<"\t"<<res_curve[2]<<"\n";
                            std::vector<float> l_curv;
                            l_curv.push_back(res_curve[0]);
                            l_curv.push_back(res_curve[1]);
                            l_curv.push_back(res_curve[2]);
                            l_curv.push_back(res_curve[3]);
                            prev_lcoeff = l_curv;
                            prev_avail_lcoeff = prev_lcoeff;
                            std::vector<cv::viz::WPolyLine> resLine = detection.getLineFromCoeffs(buf[1], prev_lcoeff, 0);
                            if(detection.getProblem()){
                                std::cout<<"\nPROBLEM 2 \n";
                                thirdOrder.push_back(prev_lline);
                            }else{
                                thirdOrder.push_back(resLine[0]);
                                prev_lline = resLine[0];
                            }
                            update_flag = 1;
                            l_curv.clear();
                        }
                        else{
                            std::cout<<"Function not present \n";
                        }

                    } catch (std::exception& e) {
                        std::cout<<"Some error \n";
                        PyErr_Print();
                    }

                    if(update_flag == 0){
                        printf("Update 0 \n");
                        if (prev_avail_lcoeff.empty()){
                            printf("Lcoeff is empty \n");
                            // /emptyLUpCount++;
                            std::vector<cv::Vec3f> zero;
                            zero.push_back(cv::Vec3f(0, 0, 1));
                            cv::Mat pointsMat = cv::Mat(static_cast<int>(zero.size()), 1, CV_32FC3, &zero[0]);
                            cv::viz::WPolyLine justLine = cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());
                            thirdOrder.push_back(justLine); //left
                            update_flag = 1;
                        }
                        else{
                            prev_lcoeff = prev_avail_lcoeff;
                            thirdOrder.push_back(prev_lline);
                        }

                    }
                    
                }
                else{
                
                    std::cout<<"\n LEFT UPDATE CHECK AGAIN \n";
                    std::cout<<" Len of 3rd "<<thirdOrder.size()<<" \n";
                    std::cout<<" contents of lprev "<<prev_lcoeff[0]<<" and "<<prev_lcoeff[1];
                    
                    int countSubs = 0;
                    std::vector<double> c( lcoeff.size(), 0.0 );
                    for ( std::vector<double>::size_type i = 0; i < lcoeff.size(); i++ )
                    {
                        if(lcoeff[i] > 0 || lcoeff[i] < 0 ){
                            countSubs++;
                        }
                       
                    }
                    if(countSubs > 1){
                        std::vector<cv::viz::WPolyLine> resLiner = detection.getLineFromCoeffs(buf[1], lcoeff, 0);
                        if(detection.getProblem()){
                             std::cout<<"\nPROBLEM 4 \n";
                             thirdOrder.push_back(prev_lline);
                        }else{
                            thirdOrder.push_back(resLiner[0]);
                            prev_lline = resLiner[0];
                        }
                        std::cout<<"\n resLiner update to prev_lline \n";
                        
                    }else{
                       std::cout<<"\n COUNTSUB IS NOT > 1 \n";
                       thirdOrder.push_back(prev_lline);
                    
                    }
                
                }

            }

            /// RIGHT
            if(rcoeff.empty()){
                std::cout<< "EMPTY .... RIGHT" << "\n";
               
                if (prev_avail_rcoeff.empty()){
                    printf("Rcoeff is empty \n");
                    // emptyRCount++;
                    std::vector<cv::Vec3f> zero;
                    zero.push_back(cv::Vec3f(0, 0, 1));
                    cv::Mat pointsMat = cv::Mat(static_cast<int>(zero.size()), 1, CV_32FC3, &zero[0]);
                    cv::viz::WPolyLine justLine = cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());
                    thirdOrder.push_back(justLine); //right
                }
                else{
                    std::cout<< "EMPTY .... RIGHT PREV" << "\n";
                    // emptyRPrevCount++;
                    prev_rcoeff = prev_avail_rcoeff;
                    std::cout<< "EMPTY RIGHT HERE " << prev_avail_rcoeff[0]<< "\t"<< prev_avail_rcoeff[1] <<"\n";
                    std::vector<cv::viz::WPolyLine> resLine = detection.getLineFromCoeffs(buf[1], prev_rcoeff, 1);
                    if(detection.getProblem()){
                        std::cout<<"\nPROBLEM 1 \n";
                        thirdOrder.push_back(prev_rline);
                    }else{
                        thirdOrder.push_back(resLine[0]);
                        prev_rline = resLine[0];
                    }
                }
            }
            else{
                std::cout<<" RIGHTPOINTCOUNT "<<rightPointCount;
                
                if(rightPointCount > 20){
                    std::cout<<"R update \n";
                    //emptyRUpCount++;
                    pArgs = PyTuple_New(5);
                    PyObject* pointsLeft = vectorToList_Float(v_left);
                    PyObject* pointsRight = vectorToList_Float(v_right);
                    PyObject* lList = NullListObj();
                    PyObject* rList = vectorToList_Float(rcoeff);
                    std::cout<<"Frame idx two : \t"<<frame_num<<"\n";
                    pValue = PyFloat_FromDouble(frame_num);
                    PyTuple_SetItem(pArgs, 0, pointsLeft);
                    PyTuple_SetItem(pArgs, 1, pointsRight);
                    PyTuple_SetItem(pArgs, 2, lList);
                    PyTuple_SetItem(pArgs, 3, rList);
                    PyTuple_SetItem(pArgs, 4, pValue);
                    if(prev_avail_rcoeff.empty()){
                            std::cout<<" PREV RCOEFF IS EMPTY!!!! \n";
                    }

                    int update_flag = 0;

                    try {
                        if(PyCallable_Check(pFunc)){
                            // Update line 
                            PyObject * pResult = PyObject_CallObject(pFunc, pArgs);
                            // updateNum ++;
                            std::vector<float> res_curve = listTupleToVector_Float(pResult);
                            std::cout<<"Got points Rorig "<<rcoeff[0]<<"\t"<<rcoeff[1]<<"\t"<<rcoeff[2]<<"\n";
                            std::cout<<"Got points "<<res_curve[3]<<"\t"<<res_curve[4]<<"\t"<<res_curve[5]<<"\n";
                            std::vector<float> r_curv;
                            r_curv.push_back(res_curve[4]);
                            r_curv.push_back(res_curve[5]);
                            r_curv.push_back(res_curve[6]);
                            r_curv.push_back(res_curve[7]);
                            prev_rcoeff = r_curv;
                            prev_avail_rcoeff = prev_rcoeff;
                            
                            std::vector<cv::viz::WPolyLine> resLine = detection.getLineFromCoeffs(buf[1], prev_rcoeff, 1);
                            if(detection.getProblem()){
                                std::cout<<"\nPROBLEM 2 \n";
                                thirdOrder.push_back(prev_rline);
                            }else{
                                thirdOrder.push_back(resLine[0]);
                                prev_rline = resLine[0];
                            }
                            update_flag = 1;
                            r_curv.clear();
                        }
                        else{
                            std::cout<<"Function not present \n";
                        }

                    } catch (std::exception& e) {
                        std::cout<<"Some error \n";
                        PyErr_Print();
                    }
                    if(update_flag == 0){
                        printf("Update r0 \n");
                        if (prev_avail_rcoeff.empty()){
                            printf("Rcoeff is empty \n");
                            std::vector<cv::Vec3f> zero;
                            zero.push_back(cv::Vec3f(0, 0, 1));
                            cv::Mat pointsMat = cv::Mat(static_cast<int>(zero.size()), 1, CV_32FC3, &zero[0]);
                            cv::viz::WPolyLine justLine = cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());
                            thirdOrder.push_back(justLine); //right
                            if(prev_avail_rcoeff.empty()){
                                std::cout<<" PREV RCOEFF IS EMPTY! \n";
                            }
                        }
                        else{
                            prev_rcoeff = prev_avail_rcoeff;
                            prev_rcoeff = rcoeff;
                            prev_avail_rcoeff = prev_rcoeff;
                            std::vector<cv::viz::WPolyLine> resLiner = detection.getLineFromCoeffs(buf[1], prev_avail_rcoeff, 1);
                            if(detection.getProblem()){
                                std::cout<<"\nPROBLEM 3 \n";
                                thirdOrder.push_back(prev_rline);
                            }else{
                                thirdOrder.push_back(resLiner[0]);
                                prev_rline = resLiner[0];
                            }
                            std::cout<<" resLiner update to prev_rline (EMPTY) \n";
                            
                            if(prev_avail_rcoeff.empty()){
                            std::cout<<" PREV RCOEFF IS EMPTY!! \n";
                            }
                        }

                    }
                    /////////////////////////////////////////////

                    
                }
                else{
                    
                    printf("\n RIGHT UPDATE CHECK AGAIN \n");

                    if(prev_avail_rcoeff.empty()){
                        std::cout<<" PREV RCOEFF IS EMPTY!!! \n";
                    }
                    std::cout<<" Len of 3rd "<<thirdOrder.size()<<" \n";
                    std::cout<<" contents of rprev "<<prev_rcoeff[0]<<" and "<<prev_rcoeff[1];
                    
                    int countSubs = 0;
                    std::vector<double> c( rcoeff.size(), 0.0 );
                    for ( std::vector<double>::size_type i = 0; i < rcoeff.size(); i++ )
                    {
                        if(rcoeff[i] > 0 || rcoeff[i] < 0 ){
                            countSubs++;
                        }
                       
                    }
                    if(countSubs > 1){
                        std::vector<cv::viz::WPolyLine> resLiner = detection.getLineFromCoeffs(buf[1], rcoeff, 1);
                        if(detection.getProblem()){
                             std::cout<<"\nPROBLEM 4 \n";
                             thirdOrder.push_back(prev_rline);
                        }else{
                            thirdOrder.push_back(resLiner[0]);
                            prev_rline = resLiner[0];
                        }
                        std::cout<<"\n resLiner update to prev_rline \n";
                        
                    }else{
                       std::cout<<"\n COUNTSUB IS NOT > 1 \n";
                       thirdOrder.push_back(prev_rline);
                    
                    }
                }
                

            }
            

        }
        
        else{
            printf("________________________data found both _______________________________%d\n");
            // BOTH 
            /////////////////////////////////////////
            // fullLRCount++;
            pArgs = PyTuple_New(5);
            PyObject* pointsLeft = vectorToList_Float(v_left);
            PyObject* pointsRight = vectorToList_Float(v_right);
            PyObject* lList = vectorToList_Float(lcoeff);
            PyObject* rList = vectorToList_Float(rcoeff);
            
            std::cout<<"FRAME NUM "<< frame_num;
            // pValue = PyFloat_FromDouble(frame_idx);
            pValue = PyFloat_FromDouble(frame_num);
            
            PyTuple_SetItem(pArgs, 0, pointsLeft);
            PyTuple_SetItem(pArgs, 1, pointsRight);
            PyTuple_SetItem(pArgs, 2, lList);
            PyTuple_SetItem(pArgs, 3, rList);
            PyTuple_SetItem(pArgs, 4, pValue);
            
            
            int update_flag = 0;

            try {
                if(PyCallable_Check(pFunc)){
                    // Update line 
                    PyObject * pResult = PyObject_CallObject(pFunc, pArgs);
                    // updateNum++;
                    std::vector<float> res_curve = listTupleToVector_Float(pResult);
                    std::cout<<"Got points lorig "<<lcoeff[0]<<"\t"<<lcoeff[1]<<"\t"<<lcoeff[2]<<"\n";
                    std::cout<<"Got points rorig "<<rcoeff[0]<<"\t"<<rcoeff[1]<<"\t"<<rcoeff[2]<<"\n";
                    std::cout<<"Got points "<<res_curve[0]<<"\t"<<res_curve[1]<<"\t"<<res_curve[2]<<"\n";
                    std::cout<<"Got points "<<res_curve[4]<<"\t"<<res_curve[5]<<"\t"<<res_curve[6]<<"\n";
                    std::vector<float> l_curv;
                    std::vector<float> r_curv;
                    l_curv = {};
                    r_curv = {};

                    l_curv.push_back(res_curve[0]);
                    l_curv.push_back(res_curve[1]);
                    l_curv.push_back(res_curve[2]);
                    l_curv.push_back(res_curve[3]);
                    r_curv.push_back(res_curve[4]);
                    r_curv.push_back(res_curve[5]);
                    r_curv.push_back(res_curve[6]);
                    r_curv.push_back(res_curve[7]);
                    prev_lcoeff = l_curv;
                    prev_avail_lcoeff = prev_lcoeff;
                    prev_rcoeff = r_curv;
                    prev_avail_rcoeff = prev_rcoeff;
                    std::vector<cv::viz::WPolyLine> resLinel = detection.getLineFromCoeffs(buf[1], prev_avail_lcoeff, 0);
                    std::vector<cv::viz::WPolyLine> resLiner = detection.getLineFromCoeffs(buf[1], prev_avail_rcoeff, 1);
                    thirdOrder.push_back(prev_lline);
                    if(detection.getProblem()){
                         std::cout<<"\nPROBLEM 5 \n";
                         thirdOrder.push_back(prev_rline);
                        }else{
                        thirdOrder.push_back(resLiner[0]);
                        prev_rline = resLiner[0];
                    }
                    prev_lline = resLinel[0];
                    update_flag = 1;
                    r_curv.clear();
                    l_curv.clear();
                    std::cout<<" clearing both curve \n";
                    if(prev_avail_rcoeff.empty()){
                        std::cout<<" PREV RCOEFF IS EMPTY!!!!! \n";
                    }
                }
                else{
                    std::cout<<"Function not present \n";
                }
                
            } catch (std::exception& e) {
                std::cout<<"Some error \n";
                PyErr_Print();
            }
            if(update_flag == 0){
                printf("Update both 0 \n");
                prev_lcoeff = prev_avail_lcoeff;
                thirdOrder.push_back(prev_lline);
                prev_rcoeff = prev_avail_rcoeff;
                thirdOrder.push_back(prev_rline);

            }

        }
        // detection.writeResultTotxttracking(prev_avail_lcoeff, 0, kittiVideoFolder);
        // detection.writeResultTotxttracking(prev_avail_rcoeff, 1, kittiVideoFolder);
}


template<typename T> void print_queue_ptr(T &q){
    while(!q->empty()){
       
        q->pop();
    }
    std::cout<<"\n";
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
    int lidar_num = 0;
    float rot_deg = 0;
    std::string direction = "front";
    
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
    
    int frameStart = 10, frameEnd = 200; // Doesnt matter, using glob inside datareader
    long long timeStamp;
    
    // ::Boundary detect
    // for kitti 16 data
    Boundary_detection detection(64, 1.125, datePath, kittiVideoFolder+"/", frameStart, frameEnd+1, false);
    // for our data
    // Boundary_detection detection(16, 1.125, datePath, kittiVideoFolder+"/", frameStart, frameEnd+1, false);
    std::cout<<"Debug6\n";
    // Virtual scan object
    FastVirtualScan virtualscan = FastVirtualScan();
    fusion::FusionController fuser;

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
    
    // Main Event loop
    int frame_idx = 0;
    std::string mem_name;

    mem_name = "Pointcloud_shared_memory_"+std::to_string(lidar_num);
    // bi::shared_memory_object::remove(mem_name.c_str());
    bi::managed_shared_memory segment_pcd(bi::open_or_create , mem_name.c_str(),155000*100); //segment name          //segment size in bytes
    std::cout<<"Constructed Point space \n";

    mem_name = "Boundary_shared_memory_"+std::to_string(lidar_num);
    // bi::shared_memory_object::remove(mem_name.c_str());
    bi::managed_shared_memory segment_bndr(bi::open_or_create , mem_name.c_str(),155000*100); //segment name          //segment size in bytes
    std::cout<<"Constructed Point space \n";

    mem_name = "Vscan_shared_memory_"+std::to_string(lidar_num);
    // bi::shared_memory_object::remove(mem_name.c_str());
    bi::managed_shared_memory segment_vscan(bi::open_or_create , mem_name.c_str(),155000*100); //segment name          //segment size in bytes
    std::cout<<"Constructed Point space \n";

    mem_name = "Lidar_update_status_"+std::to_string(lidar_num);
    // bi::shared_memory_object::remove(mem_name.c_str());
    bi::managed_shared_memory segment_status(bi::open_or_create ,mem_name.c_str(),1000);
    std::cout<<"Made Update space \n";
    
    typedef bi::allocator<float, bi::managed_shared_memory::segment_manager> ShmemAllocator;
    typedef bi::vector<float, ShmemAllocator> bVector;
    //Initialize shared memory STL-compatible allocator
    ShmemAllocator alloc_inst (segment_pcd.get_segment_manager());
    alloc_inst.allocate(36000*3);
    ShmemAllocator alloc_inst_bndr (segment_bndr.get_segment_manager());
    alloc_inst_bndr.allocate(24000*3);
    ShmemAllocator alloc_inst_vscan (segment_vscan.get_segment_manager());
    alloc_inst_vscan.allocate(24000*3);
    
    mem_name = "Pointcloud_shared_mutex_"+std::to_string(lidar_num);
    bi::interprocess_mutex *mtx_pcd = segment_pcd.find_or_construct<bi::interprocess_mutex>(mem_name.c_str())();

    // mem_name = "Boundary_shared_mutex_"+std::to_string(lidar_num);
    // bi::interprocess_mutex *mtx_pcd_bndr = segment_bndr.find_or_construct<bi::interprocess_mutex>(mem_name.c_str())();

    // mem_name = "Vscan_shared_mutex_"+std::to_string(lidar_num);
    // bi::interprocess_mutex *mtx_vscan = segment_vscan.find_or_construct<bi::interprocess_mutex>(mem_name.c_str())();
    
    mem_name = "Pointcloud_shared_cnd_"+std::to_string(lidar_num);
    bi::interprocess_condition *cnd_pcd = segment_pcd.find_or_construct<bi::interprocess_condition>(mem_name.c_str())();
    bi::scoped_lock<bi::interprocess_mutex> lock_pcd{*mtx_pcd};

    // mem_name = "Boundary_shared_cnd_"+std::to_string(lidar_num);
    // bi::interprocess_condition *cnd_pcd_bndr = segment_bndr.find_or_construct<bi::interprocess_condition>(mem_name.c_str())();
    // // bi::scoped_lock<bi::interprocess_mutex> lock_pcd_bndr{*mtx_pcd_bndr};

    // mem_name = "Vscan_shared_cnd_"+std::to_string(lidar_num);
    // bi::interprocess_condition *cnd_vscan = segment_vscan.find_or_construct<bi::interprocess_condition>(mem_name.c_str())();
    // // bi::scoped_lock<bi::interprocess_mutex> lock_vscan{*mtx_vscan};

    // bi::MutexType mutex;
    while(!viewer.wasStopped())
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

        std::cout<<"Waiting now \n";
        
        std::cout<<"Direction:  "<< direction<<"\n";
        std::vector<int> globFrameNums = detection.retrieveData(direction);
        detection.pointcloud_preprocessing(rot_vec[0]); 
        // if(direction == "back"){
        //     auto flip_rot = SensorConfig::getRotationMatrixFromTheta(180.0);
        //     detection.rotate_and_translate_multi_lidar_yaw(flip_rot);
        // }
        std::vector<std::vector<float>> &pointcloud_front = detection.get_pointcloud();
        std::vector<cv::Vec3f> buffer;
        LidarViewer::pushToBuffer(buffer, pointcloud_front);
        QVector<double> beams; 
        virtualscan.calculateVirtualScans(buffer, BEAMNUM, STEP, MINFLOOR, MAXCEILING, OBSTACLEMINHEIGHT, MAXBACKDISTANCE, 
                                        ROTATION * PI / 180.0, MINRANGE);
        virtualscan.getVirtualScan(ROADSLOPMINHEIGHT * PI / 180.0, ROADSLOPMAXHEIGHT * PI / 180.0, MAXFLOOR, MINCEILING, 
                                PASSHEIGHT, beams);
        std::pair<std::vector<std::vector<float>>, std::unordered_map<int, int>> vcan_results = getVscanResult(virtualscan, beams);
        std::vector<std::vector<cv::Vec3f>> boundary_buffer = detection.getBoundaryBuffers(rot_vec[0], trans_vec[0], vcan_results.first, vcan_results.second, kittiVideoFolder);
        std::vector<std::vector<bool>> results_front(2);
        results_front[0] = std::vector<bool>(pointcloud_front.size(), false);
        results_front[0] = detection.get_result_bool(); 
        try{
            
            //Construct a shared memory
            mem_name = "Pointcloud_shared_"+std::to_string(lidar_num);
            std::cout<<"Allocated name : "<< mem_name.c_str()<<"\n";
            bVector *bVec = segment_pcd.find_or_construct<bVector>(mem_name.c_str())(alloc_inst);//first ctor parameter
            if(bVec->size() > 0){
                bVec->clear();
            }

            mem_name = "Boundary_shared_"+std::to_string(lidar_num);
            bVector *bVec_bndr = segment_bndr.find_or_construct<bVector>(mem_name.c_str())(alloc_inst_bndr);
            if(bVec_bndr->size() > 0){
                bVec_bndr->clear();
            }

            mem_name = "Vscan_shared_"+std::to_string(lidar_num);
            bVector *bVec_vscan = segment_vscan.find_or_construct<bVector>(mem_name.c_str())(alloc_inst_vscan);
            if(bVec_vscan->size() > 0){
                bVec_vscan->clear();
            }

            std::cout<<"Allocated vector space \n";
            int *updateStatus = segment_status.find_or_construct<int>("Update_status")(99);
            std::cout<<"Constructed update signal \n";
            *updateStatus = 0;

            // std::cout << "Locking pcd\n";
            // mtx_pcd->lock();
            // std::cout << "Locking pcd bndr\n";
            // mtx_pcd_bndr->lock();
            // std::cout << "Locking vscan\n";
            // mtx_vscan->lock();
            //Insert data in the vector
            // bi::scoped_lock<MutexType> lock(mutex, bi::try_to_lock);
            // if(lock){
                std::cout<<"Pushing points  \n";
                for (int i = 0; i < pointcloud_front.size(); i++){
                    bVec->push_back(pointcloud_front[i][0]); 
                    bVec->push_back(pointcloud_front[i][1]); 
                    bVec->push_back(pointcloud_front[i][2]);
                } 
                std::cout<<"Side of bvec : "<<bVec->size()<<"\n";
                std::cout<<"Done \n";
                

                
                std::cout<<"Pushing boundary buffer  \n";
                for (int i = 0; i < boundary_buffer[1].size(); i++){
                    bVec_bndr->push_back(boundary_buffer[1][i][0]); 
                    bVec_bndr->push_back(boundary_buffer[1][i][1]); 
                    bVec_bndr->push_back(boundary_buffer[1][i][2]);
                } 
                std::cout<<"Side of bVec_bndr : "<<bVec_bndr->size()<<"\n";
                std::cout<<"Done \n";
                

                
                std::cout<<"Pushing vscan buffer \n";
                for (int i = 0; i < vcan_results.first.size(); i++){
                    bVec_vscan->push_back(vcan_results.first[i][0]); 
                    bVec_vscan->push_back(vcan_results.first[i][1]); 
                    bVec_vscan->push_back(vcan_results.first[i][2]);
                    bVec_vscan->push_back(vcan_results.first[i][3]);
                } 
                std::cout<<"Side of bVec_vscan : "<<bVec_vscan->size()<<"\n";
                std::cout<<"Done \n";
                
                // std::cout << "Unlocking pcd\n";
                // mtx_pcd->unlock();
                // std::cout << "Unlocking pcd bndr\n";
                // mtx_pcd_bndr->unlock();
                // std::cout << "unlocking vscan\n";
                // mtx_vscan->unlock();
                // std::cout << "Notify\n";
                // cnd_pcd->notify_one();
                cnd_pcd->wait(lock_pcd);      

            // }
            *updateStatus = 1; 
            
        }
        catch(...){
            bi::shared_memory_object::remove("Pointcloud_shared_memory");
            bi::shared_memory_object::remove("Boundary_shared_memory");
            std::cout<<"Some error occured \n";
            throw;
        }

        frame_idx++;
    }
    
    viewer.close();
    std::cout << "Average time per frame: " << (total_ms / (frame_idx-10)) << " ms\n";
    return 0;
}

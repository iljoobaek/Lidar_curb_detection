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
    
    char* defaultArgs[6];
    int DEBUG = 0;
    
    std::string rootGivenPath ;
    std::string dataPath;
    std::string kittiDateFolder;
    std::string kittiVideoFolder;
    std::string debug;
    std::cout<<"Debug1\n";
    defaultArgs[1] = "/home/droid/manoj_work/Lidar_curb_detection_full/source";
    defaultArgs[2] =  "/home/droid/manoj_work/Lidar_curb_detection_full/kitti_data/";
    defaultArgs[3] = "2011_09_26";
    defaultArgs[4] = "2011_09_26_drive_0013_sync";
    defaultArgs[5] = "DEBUG";
    
    int numOfVelodynes;
    if (argc < 2){
        numOfVelodynes = 6;
        rootGivenPath  = defaultArgs[1];
        dataPath = defaultArgs[2];
        kittiDateFolder = defaultArgs[3];
        kittiVideoFolder = defaultArgs[4];
        debug = defaultArgs[5];
    }
    else if (argc >= 2 && argc < 6)
    {
        std::cout<<"Debugc\n";
        rootGivenPath = argv[1]; // "/home/droid/manoj_work/Lidar_curb_detection_full/source"
        dataPath = argv[2]; //"/home/droid/manoj_work/Lidar_curb_detection_full/kitti_data/";
        kittiDateFolder = argv[3]; // "2011_09_26"
        kittiVideoFolder = argv[4]; // "2011_09_26_drive_0013_sync"
        debug = argv[5];
        std::cout<<"Debugd\n";
    } 
    else if (argc == 7) 
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
    std::cout<<"Debugo\n";
    if(debug.compare("DEBUG") == 0)
            DEBUG = 1;
    std::cout<<"Debug3\n";
    std::string evalPath = pathJoin(pathJoin(pathJoin(rootGivenPath, "evaluation/accuracy_calculation/"),  kittiDateFolder), kittiVideoFolder);
    std::string datePath = pathJoin(dataPath, kittiDateFolder);
    std::cout<<"Eval path is "<<evalPath<<"\n";
    std::cout<<"Data path is "<<dataPath<<"\n";
    std::cout<<"Date path is "<<datePath<<"\n";
    std::cout<<"Total path is "<<pathJoin(datePath, kittiVideoFolder)<<"\n";
    numOfVelodynes = 1;

    // Signal Handler for pause/resume viewer
    std::signal(SIGINT, signalHandler);
    double total_ms = 0.0;
    std::cout<<"Debug4\n";
    // // Get pcap file names
    std::vector<std::string> pcap_files = SensorConfig::getPcapFiles(); 
    
    // Create Viz3d Viewer and register callbacks
    cv::viz::Viz3d viewer( "Velodyne" );
    bool pause(false);
    LidarViewer::cvViz3dCallbackSetting(viewer, pause);
    std::cout<<"Debug5\n";
    // Get rotation and translation parameters from lidar to vehicle coordinate
    auto rot_params = SensorConfig::getRotationParams();
    auto rot_vec = SensorConfig::getRotationMatrices(rot_params);
    auto trans_vec = SensorConfig::getTranslationMatrices();
    std::cout<<"Debugk\n";
    // Boundary detection object : kitti data How many frames?
    int frameStart = 0, frameEnd = 100; // DOesnt matter, using glob inside datareader
    Boundary_detection detection(64, 1.125, datePath, kittiVideoFolder+"/", frameStart, frameEnd+1, true);
    std::cout<<"Debug6\n";
    // Virtual scan object
    FastVirtualScan virtualscan = FastVirtualScan();
    fusion::FusionController fuser;

    Py_Initialize();
    PyObject *pName, *pModule, *pDict, *pFunc, *pArgs, *pValue;
    
    std::cout<<"Debug1\n";
    // Import tracker
    pModule = PyImport_Import(PyString_FromString("tracking"));
    pDict = PyModule_GetDict(pModule);
    if (!pModule) {
        PyErr_Print();
        exit(1);
    }
    std::cout<<"Debug2\n";
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
        std::vector<std::vector<bool>> results(numOfVelodynes); 
        std::vector<std::vector<int>> results_int(numOfVelodynes); 

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

        auto buf = detection.runDetection(rot_vec[0], trans_vec[0]);
        
        // auto buf = detection.getLidarBuffers(detection.get_pointcloud(), detection.get_result_bool());
        std::vector<cv::viz::WPolyLine> thirdOrder = detection.getThirdOrderLines(buf[1]);
        results[0] = detection.get_result_bool();
        
        std::vector<std::vector<cv::Vec3f>> linePoints= detection.getLeftRightLines(buf[1]);
        
        std::vector<cv::Vec3f> pc = buf[1];

        std::vector<float> lcoeff = detection.getLeftBoundaryCoeffs();
        std::vector<float> rcoeff = detection.getRightBoundaryCoeffs();
        prev_lcoeff = lcoeff;
        prev_rcoeff = rcoeff;
        std::cout<< "HERE " << "\t";
        std::vector<float> v_full;
        std::vector<float> v_left;
        std::vector<float> v_right;

        for(size_t i=0; i<linePoints[0].size(); ++i) 
        {
            const cv::Vec3f& c = linePoints[0][i];
            v_left.push_back(c[0]);
            v_left.push_back(c[1]);
            v_left.push_back(c[2]);   
            std::cout<< v_left.size() << "\t";
        }
        std::cout<< "\n";
        for(size_t i=0; i<linePoints[1].size(); ++i) 
        {
            const cv::Vec3f& c = linePoints[1][i];
            v_right.push_back(c[0]);
            v_right.push_back(c[1]);
            v_right.push_back(c[2]);   
            std::cout<< v_right.size() << "\t";
        }
        std::cout<< "\n";

        if ( lcoeff.empty() || rcoeff.empty() ) {
            std::cout<< "EMPTY ...." << "\n";
            /// LEFT
            if(lcoeff.empty()){
                std::cout<< "EMPTY .... LEFT" << "\n";
                if (prev_avail_lcoeff.empty()){
                    printf("Lcoeff is empty \n");
                    std::vector<cv::Vec3f> zero;
                    zero.push_back(cv::Vec3f(0, 0, 1));
                    cv::Mat pointsMat = cv::Mat(static_cast<int>(zero.size()), 1, CV_32FC3, &zero[0]);
                    cv::viz::WPolyLine justLine = cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());
                    thirdOrder.push_back(justLine); //left
                    
                }
                else{
                    std::cout<< "EMPTY .... LEFT PREV" << "\n";
                    std::cout<<"LEFT SIZE: "<< prev_avail_lcoeff.size()<< "\n";
                    prev_lcoeff = prev_avail_lcoeff;
                    // std::vector<cv::viz::WPolyLine> resLine = detection.getLineFromCoeffs(buf[1], prev_lcoeff, 0); //left
                    // thirdOrder.push_back(resLine[0]); //left
                    thirdOrder.push_back(prev_lline);
                }
            }
            else{
                // Update
                /////////////////////////////////////////
                std::cout<<"Left update \n";
                pArgs = PyTuple_New(5);
                // PyObject* pointsList = vectorToList_Float(v_full);
                PyObject* pointsLeft = vectorToList_Float(v_left);
                PyObject* pointsRight = vectorToList_Float(v_right);
                // PyObject* pointsList = vectorToList_Float(v_full
                PyObject* lList = vectorToList_Float(lcoeff);
                PyObject* rList = NullListObj();
                std::cout<<"Frame idx one : \t"<<frameStart+frame_idx<<"\n";
                pValue = PyFloat_FromDouble(frameStart+frame_idx);
                
                // PyTuple_SetItem(pArgs, 0, pointsList);
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
                        std::vector<cv::viz::WPolyLine> resLine = detection.getLineFromCoeffs(buf[1], l_curv, 0); // left
                        thirdOrder.push_back(resLine[0]);
                        prev_lline = resLine[0]; 
                        update_flag = 1;
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
                /////////////////////////////////////////////

            }

            /// RIGHT
            if(rcoeff.empty()){
                std::cout<< "EMPTY .... RIGHT" << "\n";
                if (prev_avail_rcoeff.empty()){
                    printf("Rcoeff is empty \n");
                    std::vector<cv::Vec3f> zero;
                    zero.push_back(cv::Vec3f(0, 0, 1));
                    cv::Mat pointsMat = cv::Mat(static_cast<int>(zero.size()), 1, CV_32FC3, &zero[0]);
                    cv::viz::WPolyLine justLine = cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());
                    thirdOrder.push_back(justLine); //right
                }
                else{
                    std::cout<< "EMPTY .... RIGHT PREV" << "\n";
                    prev_rcoeff = prev_avail_rcoeff;
                    thirdOrder.push_back(prev_rline);
                }
            }
            else{
                // Update
                ///////////////////////////////////////// 
                std::cout<<"RIght update \n";
                pArgs = PyTuple_New(5);
                PyObject* pointsLeft = vectorToList_Float(v_left);
                PyObject* pointsRight = vectorToList_Float(v_right);
                PyObject* lList = NullListObj();
                PyObject* rList = vectorToList_Float(rcoeff);
                std::cout<<"Frame idx two : \t"<<frameStart+frame_idx<<"\n";
                pValue = PyFloat_FromDouble(frameStart+frame_idx);
                
                // PyTuple_SetItem(pArgs, 0, pointsList);
                // PyTuple_SetItem(pArgs, 1, lList);
                // PyTuple_SetItem(pArgs, 2, rList);
                // PyTuple_SetItem(pArgs, 3, pValue);
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
                        std::vector<cv::viz::WPolyLine> resLine = detection.getLineFromCoeffs(buf[1], r_curv, 1); // right
                        thirdOrder.push_back(resLine[0]);
                        prev_rline = resLine[0];
                        update_flag = 1;
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
                    }
                    else{
                        prev_rcoeff = prev_avail_rcoeff;
                        thirdOrder.push_back(prev_rline);
                    }

                }
                /////////////////////////////////////////////

            }
            

        }
        
        else{
            printf("________________________data found both _______________________________%d\n");

            /////////////////////////////////////////
    
            pArgs = PyTuple_New(5);
            PyObject* pointsLeft = vectorToList_Float(v_left);
            PyObject* pointsRight = vectorToList_Float(v_right);
            PyObject* lList = vectorToList_Float(lcoeff);
            PyObject* rList = vectorToList_Float(rcoeff);
            std::cout<<"Frame idx last : \t"<<frameStart+frame_idx<<"\n";
            pValue = PyFloat_FromDouble(frameStart+frame_idx);
            
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
                    std::vector<float> res_curve = listTupleToVector_Float(pResult);
                    std::cout<<"Got points lorig "<<lcoeff[0]<<"\t"<<lcoeff[1]<<"\t"<<lcoeff[2]<<"\n";
                    std::cout<<"Got points rorig "<<rcoeff[0]<<"\t"<<rcoeff[1]<<"\t"<<rcoeff[2]<<"\n";
                    std::cout<<"Got points "<<res_curve[0]<<"\t"<<res_curve[1]<<"\t"<<res_curve[2]<<"\n";
                    std::cout<<"Got points "<<res_curve[4]<<"\t"<<res_curve[5]<<"\t"<<res_curve[6]<<"\n";
                    std::vector<float> l_curv;
                    std::vector<float> r_curv;
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
                    std::vector<cv::viz::WPolyLine> resLinel = detection.getLineFromCoeffs(buf[1], prev_lcoeff, 0);
                    std::vector<cv::viz::WPolyLine> resLiner = detection.getLineFromCoeffs(buf[1], prev_rcoeff, 1);
                    thirdOrder.push_back(resLinel[0]);
                    thirdOrder.push_back(resLiner[0]);
                    prev_lline = resLinel[0];
                    prev_rline = resLiner[0];
                    update_flag = 1;
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
                // std::vector<cv::Vec3f> zero;
                // zero.push_back(cv::Vec3f(0, 0, 1));
                // cv::Mat pointsMat = cv::Mat(static_cast<int>(zero.size()), 1, CV_32FC3, &zero[0]);
                // cv::viz::WPolyLine justLine = cv::viz::WPolyLine(pointsMat, cv::viz::Color::gold());
                // thirdOrder.push_back(justLine);
                prev_lcoeff = prev_avail_lcoeff;
                // std::vector<cv::viz::WPolyLine> resLinel = detection.getLineFromCoeffs(buf[1], prev_lcoeff, 0); // left
                // thirdOrder.push_back(resLinel[0]);
                thirdOrder.push_back(prev_lline);
                prev_rcoeff = prev_avail_rcoeff;
                // std::vector<cv::viz::WPolyLine> resLiner = detection.getLineFromCoeffs(buf[1], prev_rcoeff, 1); //right
                // thirdOrder.push_back(resLiner[0]);
                thirdOrder.push_back(prev_rline);

            }
            /////////////////////////////////////////////


            // viewer.spinOnce();
            // continue;

        }

    
        cv::viz::WPolyLine gtLine = generateGTWPolyLine(evalPath, frameStart+frame_idx); //  frameStart+frame_idx
        std::cout << " Reached here \n";
        
        std::vector<std::vector<cv::Vec3f>> temp = fuser.rotatedPointParametersVizualize(buf[1]);
//         cv::viz::WText3D firstText = cv::viz::WText3D("First Point" , fuser.firstPoint) 
        
        LidarViewer::updateViewerFromBuffers(buffers, results, viewer, res, thirdOrder, gtLine, temp[0], temp[1]);
//         LidarViewer::showText(firstText)
        
        
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

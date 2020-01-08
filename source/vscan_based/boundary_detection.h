#include <iostream>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <stack>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <Python.h>

#include <Eigen/Dense>

#include <boost/interprocess/managed_shared_memory.hpp>
#include <boost/interprocess/containers/vector.hpp>
#include <boost/interprocess/allocators/allocator.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <boost/interprocess/sync/named_mutex.hpp>

#ifndef CV2_H
#define CV2_H
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#endif

#include "data_reader.h"
#include "Fusion.cpp"

#define PI 3.14159265
#define THETA_R 0.00356999
#define MIN_CURB_HEIGHT 0.05

class Object_detection {
public:
    Object_detection() {
        // path of the virtual env
        setenv("PYTHONHOME", "/home/rtml/Lidar_curb_detection/source/lidar_based/curb_detection_cpp/env/", true);
        Py_Initialize();
        if ( !Py_IsInitialized() ){
            std::cerr << "Initialize failed\n";
        }
        else std::cout << "Python interpreter initialized\n";
        PyRun_SimpleString("import sys");
        PyRun_SimpleString("sys.path.append('./')");
        this->pName = PyString_FromString("detector");
	    this->pModule = PyImport_Import(this->pName);
	    if ( !this->pModule ){
	    	std::cerr << "Can't find Module\n";
	    	PyErr_Print();
	    }
        this->python_class = PyObject_GetAttrString(this->pModule, "ObjectDetector");
        if ( !this->python_class) std::cerr <<"can't get python class [ObjectDetector]\n";
        if (PyCallable_Check(python_class)) {
            std::cout << "Instatiate python class object\n";
            object = PyObject_CallObject(python_class, nullptr);
        }
        else {
            std::cerr <<"can't instatiate python class [ObjectDetector]\n";
        }
        std::cout << "------------------------------------------------------\n";
        this->img_width = 1280;
        this->img_height = 1024;
        this->ROI_height = 400;
        this->ROI_offset_y = 200;
    }
    
    ~Object_detection() {
        Py_DECREF(this->pName);
        Py_DECREF(this->pModule);
        Py_DECREF(this->python_class);
        Py_DECREF(this->object);
        Py_Finalize();
        std::cout << "Close Python interpreter\n";
    }

    PyObject* call_method(char *method, std::string filename) {
    	PyObject* res;
        res = PyObject_CallMethod(this->object, method, "(s)", filename.c_str());
        if (!res) PyErr_Print();
        return res;
    }

    std::vector<float> listTupleToVector(PyObject *data_in) {
        std::vector<float> data;
        if (PyTuple_Check(data_in)) {
            for (Py_ssize_t i = 0; i < PyTuple_Size(data_in); i++) {
                PyObject* value = PyTuple_GetItem(data_in, i);
                data.push_back( PyFloat_AsDouble(value) );
            }
        }
        else {
            if (PyList_Check(data_in)) {
                for (Py_ssize_t i = 0; i < PyList_Size(data_in); i++) {
                    PyObject* value = PyList_GetItem(data_in, i);
                    data.push_back( PyFloat_AsDouble(value) );
                }
            }           
            else throw std::logic_error("Passed PyObject pointer is not a list or tuple."); 
        }
        return data;
    }

    int img_width, img_height, ROI_height, ROI_offset_y;

private:
    PyObject *pName, *pModule, *python_class, *object;
};

typedef boost::interprocess::allocator<cv::Vec3f, boost::interprocess::managed_shared_memory::segment_manager>  ShmemAllocator;
typedef boost::interprocess::vector<cv::Vec3f, ShmemAllocator> radar_shared;

class Boundary_detection 
{
private:
    enum class ScanDirection 
    {
        CLOCKWISE,
        COUNTER_CLOCKWISE
    };
    enum class DataType
    {
        LIDAR,
        LIDAR_CAMERA,
        LIDAR_CAMERA_RADAR
    };
public:
    Boundary_detection(int numOfScan, float sensor_height, std::string root_path, std::string data_folder, int start, int end, bool isDownSample): 
                        dataReader(root_path, data_folder, start, end, isDownSample), 
                        tilted_angle(tilted_angle), sensor_height(sensor_height),
                        currentFrameIdx(start), root_path(root_path), data_folder(data_folder)
    {
        num_of_scan = isDownSample ? numOfScan / 4 : numOfScan;
        ranges = std::vector<std::vector<int>>(num_of_scan*2, std::vector<int>(2));
        angles_16 = {-15.0, 1.0, -13.0, 3.0, -11.0, 5.0, -9.0, 7.0,
                        -7.0, 9.0, -5.0, 11.0, -3.0, 13.0, -1.0, 15.0};
        //timedFunction(std::bind(&Boundary_detection::expose, this), 100);
        fuser = fusion::FusionController();
        object_detector = std::unique_ptr<Object_detection>(new Object_detection());
    } 
    Boundary_detection(int numOfScan, float sensor_height, std::string data_path): 
                        dataReader(data_path), 
                        tilted_angle(tilted_angle), sensor_height(sensor_height) 
    {
        num_of_scan = numOfScan;
        ranges = std::vector<std::vector<int>>(num_of_scan*2, std::vector<int>(2));
        angles_16 = {-15.0, 1.0, -13.0, 3.0, -11.0, 5.0, -9.0, 7.0,
                        -7.0, 9.0, -5.0, 11.0, -3.0, 13.0, -1.0, 15.0};
        //timedFunction(std::bind(&Boundary_detection::expose, this), 100);
        fuser = fusion::FusionController();
    }

    bool isRun();
    void retrieveData();
    void pointcloud_preprocessing(const cv::Mat &rot);
    std::vector<std::vector<cv::Vec3f>> runDetection(const cv::Mat &rot, const cv::Mat &trans);
    std::vector<std::vector<float>>& get_pointcloud();
    std::vector<int> get_result();
    std::vector<bool> get_result_bool();
    std::vector<std::vector<cv::Vec3f>> getLidarBuffers(const std::vector<std::vector<float>> &pointcloud, const std::vector<bool> &result);
    std::vector<cv::viz::WPolyLine> getThirdOrderLines(std::vector<cv::Vec3f> &buf); 
    std::vector<float> getLeftBoundaryCoeffs();
    std::vector<float> getRightBoundaryCoeffs();
    void writeResultTotxt(const std::vector<float> &boundaryCoeffs, int leftRight);

private:
    void rotate_and_translate_multi_lidar_yaw(const cv::Mat &rot);
    void max_height_filter(float max_height);
    void rearrange_pointcloud();

    std::vector<float> get_dist_to_origin();
    float dist_between(const std::vector<float> &p1, const std::vector<float> &p2);
    std::vector<bool> continuous_filter(int scan_id);
    float get_angle(const std::vector<float> &v1, const std::vector<float> &v2);
    std::vector<float> direction_change_filter(int scan_id, int k, float angle_thres=150.0f);
    std::vector<bool> local_min_of_direction_change(int scan_id);
    std::vector<int> elevation_filter(int scan_id);
    void edge_filter_from_elevation(int scan_id, const std::vector<int> &elevation, std::vector<bool> &edge_start, std::vector<bool> &edge_end);
    float distance_to_line(cv::Point2f p1, cv::Point2f p2);

    std::string get_filename_image(std::string root_dir, std::string folder, int frame_idx);
    std::vector<std::vector<float>> get_image_points();
    bool is_in_bounding_box(const std::vector<float> &point, const std::vector<std::vector<int>> &bounding_boxes);
    bool find_objects_from_image(std::string filename, cv::Mat &img);

    void find_boundary_from_half_scan(int scan_id, int k, bool masking);

    void reset();    

    void timedFunction(std::function<void(void)> func, unsigned int interval);
    void expose();
    
    boost::interprocess::named_mutex 
            mem_mutex{
                boost::interprocess::open_or_create, 
                "radar_mutex"
            };
    
private:
    // Radar
    bool firstRun = true;
    bool secondRun = false;
    fusion::FusionController fuser;

    // Lidar
    DataReader::LidarDataReader dataReader;
    std::string root_path;
    std::string data_folder;

    // Camera
    std::vector<std::vector<float>> lidar_to_image = {{6.07818353e+02, -7.79647962e+02, -8.75258198e+00, 2.24308511e+01},
                                                      {5.12565990e+02, 1.31878337e+01, -7.70608644e+02, -1.69836140e+02},
                                                      {9.99862028e-01, -8.56083140e-03, 1.42350786e-02, 9.02290525e-03}};
    // std::vector<std::vector<float>> lidar_to_image_kitti = {{7.45484183e+00, -0.00000000e+00, -4.32854604e-01, -0.00000000e+00},
    //                                                   {0.00000000e+00, 7.19218909e-01, -2.45532038e+02, -0.00000000e+00},
    //                                                   {0.00000000e+00,  0.00000000e+00,  1.48075500e-02, -0.00000000e+00}};

    std::unique_ptr<Object_detection> object_detector;
    int currentFrameIdx;

    int num_of_scan;
    float tilted_angle;
    float sensor_height;
    std::vector<float> angles_16;
    
    std::vector<std::vector<float>> pointcloud;
    std::vector<std::vector<float>> pointcloud_raw;
    std::vector<int> index_mapping;
    std::vector<std::vector<int>> ranges;
    std::vector<cv::Vec3f> radar_pointcloud;
    std::vector<float> dist_to_origin;

    std::vector<bool> is_boundary;
    std::vector<int> is_boundary_int;

    std::vector<bool> is_continuous;
    std::vector<bool> is_elevating;
    std::vector<bool> is_changing_angle;
    std::vector<bool> is_local_min;
    std::vector<bool> is_edge_start;
    std::vector<bool> is_edge_end;

    std::vector<bool> is_boundary_masking;
    std::vector<bool> is_objects;
};

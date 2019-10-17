#include <iostream>
#include <memory>
#include <vector>
#include <string>
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

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include "VelodyneCapture.h"

#include "GRANSAC.hpp"
#include "LineModel.hpp"

#include "Fusion.cpp"

#define PI 3.14159265
#define THETA_R 0.00356999
#define MIN_CURB_HEIGHT 0.05
#define USE_OBJECT_MASKING true

using std::cout;
using std::endl;
using std::vector;
using std::string;
using namespace std::chrono;

class Object_detection {
public:
    Object_detection() {
        // path of the virtual env
        setenv("PYTHONHOME", "/home/rtml/Lidar_curb_detection/source/lidar_based/curb_detection_cpp/env/", true);
        Py_Initialize();
        if ( !Py_IsInitialized() ){
            std::cerr << "Initialize failed\n";
        }
        else cout << "Python interpreter initialized\n";
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
        cout << "------------------------------------------------------\n";
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
        cout << "Close Python interpreter\n";
    }

    PyObject* call_method(char *method, string filename) {
    	PyObject* res;
        res = PyObject_CallMethod(this->object, method, "(s)", filename.c_str());
        if (!res) PyErr_Print();
        return res;
    }

    vector<float> listTupleToVector(PyObject *data_in) {
        vector<float> data;
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

class Boundary_detection {
public:
    Boundary_detection(string dir, int id, float tilted_angle, float sensor_height): directory(dir), frame_id(id), num_of_scan(16) {
        this->ranges = std::vector<std::vector<int>>(32, std::vector<int>(2));
        this->tilted_angle = tilted_angle;
        this->sensor_height = sensor_height;
        this->angles = {-15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0,
                        1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0};
        //timedFunction(std::bind(&Boundary_detection::expose, this), 100);
        if (dir.find(".pcap") != string::npos) this->isPCAP = true;
        else this->isPCAP = false;
        #if USE_OBJECT_MASKING
        this->object_detector = std::unique_ptr<Object_detection>(new Object_detection());
        #endif
        this->get_calibration();
        this->fuser = fusion::FusionController();
    } 
    
    void laser_to_cartesian(std::vector<velodyne::Laser> &lasers);
    std::vector<std::vector<float>> read_bin(string filename);
    void rotate_and_translate();
    void max_height_filter(float max_height);
    void reorder_pointcloud();
    void rearrange_pointcloud();
    void rearrange_pointcloud_unrotated();
    void rearrange_pointcloud_sort();
    void pointcloud_preprocessing();
    
    float dist_between(const std::vector<float> &p1, const std::vector<float> &p2);
    std::vector<float> get_dist_to_origin();
    std::vector<float> get_theoretical_dist();
    std::vector<bool> continuous_filter(int scan_id);
    float get_angle(const std::vector<float> &v1, const std::vector<float> &v2);
    std::vector<float> direction_change_filter(int scan_id, int k, float angle_thres=150.0f);
    std::vector<bool> local_min_of_direction_change(int scan_id);
    std::vector<int> elevation_filter(int scan_id);
    void edge_filter_from_elevation(int scan_id, const std::vector<int> &elevation, std::vector<bool> &edge_start, std::vector<bool> &edge_end);
    std::vector<bool> obstacle_extraction(int scan_id);
    std::vector<cv::Point2f> run_RANSAC(int side, int max_per_scan=10);
    float distance_to_line(cv::Point2f p1, cv::Point2f p2);

    void find_boundary_from_half_scan(int scan_id, int k, bool masking);
    std::vector<bool> run_detection(bool vis=false);

    bool get_calibration(string filename="calibration.yaml");
    std::vector<std::vector<float>> get_image_points();
    bool is_in_bounding_box(const std::vector<float> &point, const std::vector<std::vector<int>> &bounding_boxes);
    bool find_objects_from_image(string filename, cv::Mat &img);

    string get_filename_pointcloud(const string &root_dir, int frame_idx);
    string get_filename_image(const string &root_dir, int frame_idx);
    void print_pointcloud(const std::vector<std::vector<float>> &pointcloud);

    void reset();    
    std::vector<std::vector<float>>& get_pointcloud();
    std::vector<bool>& get_result();

    std::vector<std::vector<cv::Vec3f>> getLidarBuffers(const std::vector<std::vector<float>> &pointcloud, const std::vector<bool> &result);
    void timedFunction(std::function<void(void)> func, unsigned int interval);
    void expose();
    
    boost::interprocess::named_mutex 
            mem_mutex{
                boost::interprocess::open_or_create, 
                "radar_mutex"
            };

private:
    std::unique_ptr<Object_detection> object_detector;
    bool isPCAP;
    bool firstRun = true;
    bool secondRun = false;
    fusion::FusionController fuser;
    string directory;
    int frame_id;
    int num_of_scan;
    float tilted_angle;
    float sensor_height;
    std::vector<float> angles;
    std::vector<float> dist_to_origin;
    std::vector<float> theoretical_dist;
    std::vector<std::vector<float>> pointcloud;
    std::vector<std::vector<float>> pointcloud_unrotated;
    std::vector<bool> is_boundary;
    std::vector<bool> is_boundary_masking;
    std::vector<bool> is_continuous;
    std::vector<bool> is_elevating;
    std::vector<bool> is_changing_angle;
    std::vector<bool> is_local_min;
    std::vector<bool> is_edge_start;
    std::vector<bool> is_edge_end;
    std::vector<bool> is_obstacle;
    std::vector<bool> is_objects;
    std::vector<std::vector<int>> ranges;
    std::vector<std::vector<float>> lidar_to_image;
    std::vector<cv::Vec3f> radar_pointcloud;
};

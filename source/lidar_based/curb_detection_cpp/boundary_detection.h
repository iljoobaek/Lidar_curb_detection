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

#include <Eigen/Dense>

#include "VelodyneCapture.h"

#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include "GRANSAC.hpp"
#include "LineModel.hpp"

#include <Python.h>

#define PI 3.14159265
#define THETA_R 0.00356999
#define MIN_CURB_HEIGHT 0.05

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

    PyObject* call_method(char* method, string filename) {
    	PyObject* res;
        res = PyObject_CallMethod(this->object, method, "(s)", filename.c_str());
        if (!res) PyErr_Print();
        return res;
    }

    vector<float> listTupleToVector(PyObject* data_in) {
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

class Boundary_detection {
public:
    Boundary_detection(string dir, int id, float tilted_angle, float sensor_height): directory(dir), frame_id(id), num_of_scan(16) {
        this->ranges = vector<vector<int>>(32, vector<int>(2));
        this->tilted_angle = tilted_angle;
        this->sensor_height = sensor_height;
        this->angles = {-15.0, -13.0, -11.0, -9.0, -7.0, -5.0, -3.0, -1.0,
                        1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0};
        if (dir.find(".pcap") != string::npos) this->isPCAP = true;
        else this->isPCAP = false;
        this->img_width = 1280;
        this->img_height = 1024;
        this->object_detector = new Object_detection();
        this->get_calibration();
    } 
    
    void laser_to_cartesian(std::vector<velodyne::Laser>& lasers);
    vector<vector<float>> read_bin(string filename);
    void rotate_and_translate();
    void max_height_filter(float max_height);
    void reorder_pointcloud();
    void rearrange_pointcloud();
    void rearrange_pointcloud_sort();
    void pointcloud_preprocessing();
    
    float dist_between(const vector<float>& p1, const vector<float>& p2);
    vector<float> get_dist_to_origin();
    vector<float> get_theoretical_dist();
    vector<bool> continuous_filter(int scan_id);
    float get_angle(const vector<float>& v1, const vector<float>& v2);
    vector<float> direction_change_filter(int scan_id, int k, float angle_thres=150.0f);
    vector<bool> local_min_of_direction_change(int scan_id);
    vector<int> elevation_filter(int scan_id);
    void edge_filter_from_elevation(int scan_id, const vector<int>& elevation, vector<bool>& edge_start, vector<bool>& edge_end);
    vector<bool> obstacle_extraction(int scan_id);
    std::vector<cv::Point2f> run_RANSAC(int side, int max_per_scan=10);
    float distance_to_line(cv::Point2f p1, cv::Point2f p2);

    void find_boundary_from_half_scan(int scan_id, int k);
    vector<bool> run_detection(bool vis=false);

    bool get_calibration(string filename="calibration.yaml");
    vector<vector<float>> get_image_points();
    bool is_in_bounding_box(const vector<float>& point, const vector<vector<int>>& bounding_boxes);
    bool find_objects_from_image(string filename, cv::Mat& img);

    string get_filename_pointcloud(const string& root_dir, int frame_idx);
    string get_filename_image(const string& root_dir, int frame_idx);
    void print_pointcloud(const vector<vector<float>>& pointcloud);

    void reset();    
    vector<vector<float>>& get_pointcloud();
    vector<bool>& get_result();

private:
    Object_detection* object_detector;
    bool isPCAP;
    string directory;
    int frame_id;
    int num_of_scan;
    float tilted_angle;
    float sensor_height;
    vector<float> angles;
    vector<float> dist_to_origin;
    vector<float> theoretical_dist;
    vector<vector<float>> pointcloud;
    vector<vector<float>> pointcloud_unrotated;
    vector<bool> is_boundary;
    vector<bool> is_continuous;
    vector<bool> is_elevating;
    vector<bool> is_changing_angle;
    vector<bool> is_local_min;
    vector<bool> is_edge_start;
    vector<bool> is_edge_end;
    vector<bool> is_obstacle;
    vector<bool> is_objects;
    vector<vector<int>> ranges;
    vector<vector<float>> lidar_to_image;
    int img_width, img_height;
};

#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <sstream>

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
        Py_Initialize();
        if ( !Py_IsInitialized() ){
            std::cerr << "Initialize failed\n";
        }
        else cout << "Python interpreter initialized\n";
        // PySys_SetPath("./env/bin");
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
    }
    ~Object_detection() {
        Py_DECREF(this->pName);
        Py_DECREF(this->pModule);
        Py_DECREF(this->python_class);
        Py_DECREF(this->object);
        Py_Finalize();
        cout << "Close Python interpreter\n";
    }
    void call_method(char* method, int idx) {
    	PyObject* value2 = PyObject_CallMethod(this->object, method, "(s)", std::to_string(idx).c_str());
    }

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
        this->object_detector = new Object_detection();
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
    vector<bool> is_boundary;
    vector<bool> is_continuous;
    vector<bool> is_elevating;
    vector<bool> is_changing_angle;
    vector<bool> is_local_min;
    vector<bool> is_edge_start;
    vector<bool> is_edge_end;
    vector<bool> is_obstacle;
    vector<vector<int>> ranges;
};
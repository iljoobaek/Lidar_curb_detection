#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <chrono>

#include <Eigen/Dense>

#define THETA_R 0.00356999

using std::cout;
using std::endl;
using std::vector;
using std::string;
using namespace std::chrono;

// vector<float> get_dist_to_origin(const vector<vector<float>>& pointcloud);
// vector<bool> continuous_filter(const vector<vector<float>>& pointcloud, const vector<float>& dist_to_origin);
// vector<vector<float>> read_bin(string filename);
// void print_pointcloud(vector<vector<float>>& pointcloud);

class Boundary_detection {
public:
    Boundary_detection(string fn): filename(fn) {
        this->pointcloud = read_bin(fn);
    } 
    
    float dist_between(const vector<float>& p1, const vector<float>& p2);
    vector<float> get_dist_to_origin(const vector<vector<float>>& pointcloud);
    vector<bool> continuous_filter(const vector<vector<float>>& pointcloud, const vector<float>& dist_to_origin);
    vector<vector<float>> read_bin(string filename);
    void print_pointcloud(vector<vector<float>>& pointcloud);
    vector<vector<float>>& get_pointcloud();

private:
    string filename;
    vector<vector<float>> pointcloud;
};
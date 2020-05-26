// Modified from pcl library example: "http://pointclouds.org/documentation/tutorials/pcl_visualizer.php"
/* \author Geoffrey Biggs */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <thread>
#include <cmath>
#include <unordered_set>

#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/point_picking_event.h> // for 3D point picking event 

#include <Eigen/Dense>
#define PI 3.14159265

#include <nlohmann/json.hpp>
#include <experimental/filesystem>

namespace fs = std::experimental::filesystem;
using namespace Eigen;
using namespace std::chrono_literals;
using json = nlohmann::json;

std::vector<float> getThirdOrderPolynomials(const std::vector<std::vector<float>> &points) 
{
    std::vector<float> boundaryCoeffs;

    // Calculate third order polynomials by linear least square 
    MatrixXf A(points.size(), 4);
    VectorXf b(points.size());
    for (int i = 0; i < points.size(); i++) 
    {
        A(i, 0) = std::pow(points[i][1], 3);
        A(i, 1) = std::pow(points[i][1], 2);
        A(i, 2) = std::pow(points[i][1], 1);
        A(i, 3) = 1.0f;
    }
    for (int i = 0; i < points.size(); i++) 
    {
        b(i) = points[i][0];
    }
    
    // VectorXf solution = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
    VectorXf solution = A.colPivHouseholderQr().solve(b);
    std::cout << "The least-squares solution is:\n" << solution << std::endl;
    for (int i = 0; i < 4; i++)
    {
        boundaryCoeffs.push_back(solution(i));
    }
    return boundaryCoeffs;
}

void saveCoeffs(const std::string &filename, const std::vector<float> &boundaryCoeffs, const std::vector<std::vector<float>> &pointcloud)
{
    std::stringstream ss;
    assert(boundaryCoeffs.size() == 4);
    ss << boundaryCoeffs[0] << " " << boundaryCoeffs[1] << " " << boundaryCoeffs[2] << " " << boundaryCoeffs[3] << "\n";    

    if (boundaryCoeffs[0] == 0.0f && boundaryCoeffs[1] == 0.0f && boundaryCoeffs[2] == 0.0f && boundaryCoeffs[3] == 0.0f) 
    {
        return;
    }        

    std::vector<float> y;
    for (auto &point : pointcloud)
    {
        y.push_back(point[1]);
    }
    auto minmaxY = std::minmax_element(y.begin(), y.end());
    std::cout << *minmaxY.first << " " << *minmaxY.second << std::endl;

    std::ofstream file;
    file.open(filename);
    file << ss.str();
    for (int i = *minmaxY.first*100; i <= *minmaxY.second*100; i++)
    {
        float x = boundaryCoeffs[3] + boundaryCoeffs[2]*i/100. + boundaryCoeffs[1]*std::pow(i/100., 2) + boundaryCoeffs[0]*std::pow(i/100., 3);
        std::stringstream ss2;
        ss2 << x << " " << (float)i/100. << "\n";
        file << ss2.str();
    }
    file.close();
}

std::vector<std::vector<float>> readPoints(json &jsonObject)
{
    std::vector<std::vector<float>> points;
    for (json::iterator it = jsonObject.begin(); it != jsonObject.end(); ++it) 
    {
        std::vector<float> point(2);
        double x = (*it).at("x"), y = (*it).at("y"), z = (*it).at("z");
        // *** Check the coordinates ***
        point[0] = -x;
        point[1] = z;
        points.push_back(point);
    }
    return points;
}

std::string getOutputFilename(const std::string &outDir, const std::string &path)
{
    int frameIdx = 0;
    std::string fn = outDir + "/gt_";
    if (path.find("(") != std::string::npos)
    {
        int i = path.find("(") + 1;
        while (i < path.size() && std::isdigit(path[i]))
        {
            frameIdx = frameIdx * 10 + (path[i]-'0');
            i++;
        }
    }
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(10) << frameIdx;
    fn += ss.str();
    if (path.find("left") != std::string::npos)
    {
        fn += "_l.txt";
    }
    else
    {
        fn += "_r.txt";
    }
    return fn;
}

void getCoeffs(const std::string &filePath, const std::string &outDir)
{
    // Put data in json object
    std::ifstream file(filePath);
    json j;
    file >> j;
    file.close();
    auto points = readPoints(j);
    auto coeffs = getThirdOrderPolynomials(points);
    std::string fileOut = getOutputFilename(outDir, filePath);
    saveCoeffs(fileOut, coeffs, points);
}

int main (int argc, char** argv)
{
    std::cout << "----------------------\n";   
    std::cout << "json parser\n";   
    std::cout << "----------------------\n";   

    // name of the folder
    // structure is like /FOLDER_NAME
    //                      /left
    //                         files
    //                         ...   
    //                      /right/* 
    //                         files
    //  
    std::vector<std::string> rootDirs;                        
    rootDirs.push_back("0051");
    rootDirs.push_back("0048");
    
    for (auto &rootDir : rootDirs)
    {
        std::string leftDir = rootDir + "/left";
        std::string rightDir = rootDir + "/right";
        std::string outDir = rootDir + "_gt";

        // create output directory
        fs::create_directories(outDir);

        for (const auto &entry : fs::directory_iterator(leftDir))
        {
            std::string path = entry.path();
            std::cout << path << std::endl;
            getCoeffs(path, outDir);
        }
        for (const auto &entry : fs::directory_iterator(rightDir))
        {
            std::string path = entry.path();
            std::cout << path << std::endl;
            getCoeffs(path, outDir);
        }
    }
}

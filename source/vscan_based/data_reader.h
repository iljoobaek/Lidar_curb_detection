#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include <experimental/filesystem>

// Include VelodyneCapture Header
#include "VelodyneCapture.h"
namespace fs = std::experimental::filesystem;

namespace DataReader
{
class LidarDataReader
{
    enum class DataType 
    {
        PCAP,
        BINARIES,
        ROSBAG
    };
public:
    LidarDataReader(std::string fn) 
    {
        if (fn.find(".bag") != std::string::npos)
        {
            type = DataType::ROSBAG; 
        }
        else if (fn.find(".pcap") != std::string::npos)
        {
            type = DataType::PCAP;
            capture = new velodyne::VLP16Capture(fn); 
        }
    }
    LidarDataReader(std::string fn, int start, int end) : filename(fn)
    {
        type = DataType::BINARIES;
        for (int i = start; i < end; i++)
        {
            binaryFiles.push_back(getBinaryFile(i, fn));
        } 
    }
    ~LidarDataReader() {}
private:
    void laser_to_cartesian(std::vector<velodyne::Laser> &lasers, std::vector<std::vector<float>> &pointcloud, float theta, cv::Mat &rot, cv::Mat &trans) 
    {
        pointcloud.clear();
        pointcloud.reserve(lasers.size());
        for (int i = 0; i < lasers.size(); i++) 
        {
            const double distance = static_cast<double>( lasers[i].distance );
            const double azimuth  = lasers[i].azimuth  * CV_PI / 180.0;
            const double vertical = lasers[i].vertical * CV_PI / 180.0;
            float x = static_cast<float>( ( distance * std::cos( vertical ) ) * std::sin( azimuth ) );
            float y = static_cast<float>( ( distance * std::cos( vertical ) ) * std::cos( azimuth ) );
            float z = static_cast<float>( ( distance * std::sin( vertical ) ) );
            
            if( x == 0.0f && y == 0.0f && z == 0.0f ) continue;

            x /= 100.0, y /= 100.0, z /= 100.0;
            float intensity = static_cast<float>(lasers[i].intensity);
            float ring = static_cast<float>(lasers[i].id);
            float dist = std::sqrt(x * x + y * y + z * z);
            if (dist < 0.9f) continue;
            float azimuth_rot = static_cast<float>(lasers[i].azimuth) + theta;
            if (azimuth_rot >= 360.0f) 
            {
                azimuth_rot -= 360.0f;
            }
            else if (azimuth_rot < 0.0f) 
            {
                azimuth_rot += 360.0f;
            }
            pointcloud.push_back({x, y, z, intensity, ring, dist, azimuth_rot}); // Write to pointcloud
        }
    }
    std::vector<std::vector<float>> readFromBinary(std::string &filename)
    {
        std::vector<std::vector<float>> pointcloud;
        int32_t num = 1000000;
        float *data = (float *)malloc(num * sizeof(float));
        float *px = data, *py = data + 1, *pz = data + 2, *pi = data + 3, *pr = data + 4;

        FILE *stream = fopen(filename.c_str(), "rb");
        num = fread(data, sizeof(float), num, stream) / 5;
        for (int32_t i = 0; i < num; i++)
        {
            float dist = std::sqrt((*px) * (*px) + (*py) * (*py) + (*pz) * (*pz));
            float theta = std::atan2(*py, *px) * 180.0f / CV_PI;
            if (dist > 0.9f && *px >= 0.0f)
                pointcloud.push_back({*px, *py, *pz, *pi, *pr, dist, theta});
            px += 5, py += 5, pz += 5, pi += 5, pr += 5;
        }
        fclose(stream);
        std::cout << "Read in " << pointcloud.size() << " points\n";
        return pointcloud;
    }
    void rotateAndTranslate(std::vector<std::vector<float>> &pointcloud, float tilted_angle, float sensor_height)
    {
        // rotation matrix along x
        // [1,           0,           0]
        // [0,  cos(theta), -sin(theta)]
        // [0,  sin(theta),  cos(theta)]

        // rotation matrix along y
        // [cos(theta),   0, sin(theta)]
        // [0,            1,          0]
        // [-sin(theta),  0, cos(theta)]
        float theta = tilted_angle * CV_PI / 180.0f;
        // cout << "[ "<< std::cos(theta) << " " << 0.0f << " " << std::sin(theta) << "\n";
        // cout << 0.0f << " " << 1.0f << " " << 0.0f << "\n";
        // cout << -std::sin(theta) << " " << 0.0f << " " << std::cos(theta) << " ]"<< "\n";
        
        for (auto &point : pointcloud)
        {
            float x = point[0] * std::cos(theta) + point[2] * std::sin(theta);
            float z = point[0] * (-std::sin(theta)) + point[2] * std::cos(theta) + sensor_height;
            point[0] = x;
            point[2] = z;
            // Rotate along z axis to match the coordinates from pcap / velodyne capture
            float xx = point[0] * std::cos(CV_PI / 2) + point[1] * (-std::sin(CV_PI / 2));
            float yy = point[0] * std::sin(CV_PI / 2) + point[1] * std::cos(CV_PI / 2);
            point[0] = xx;
            point[1] = yy;
        }
    }

public:
    std::string getBinaryFile(int frame_idx, std::string root_dir="20190828124709/")
    {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(10) << frame_idx;
        std::string filename = "/home/rtml/lidar_radar_fusion_curb_detection/data/" + root_dir + "velodyne_points/data/" + ss.str() + ".bin";
        return filename;
    }
    bool isRun()
    {
        if (type == DataType::BINARIES)
        {
            if (currentFrame < binaryFiles.size())
            {
                return true;
            }
            else 
            {
                return false;
            } 
        }
        else if (type == DataType::PCAP)
        {
            return capture->isRun();
        }
        return false;
    }
    void operator >>(std::vector<std::vector<float>> &pointcloud) 
    {
        if (type == DataType::BINARIES)
        {
            pointcloud = readFromBinary(binaryFiles[currentFrame++]);
        }
        else if (type == DataType::PCAP)
        {
            std::vector<velodyne::Laser> lasers;
            *capture >> lasers;
            laser_to_cartesian(lasers, pointcloud, theta, rot, trans);
        }
    }

private:
    std::string filename;
    // Sensor info
    float theta;
    cv::Mat rot, trans;
    DataType type;
    // For pcap file by VelodyneCapture
    velodyne::VLP16Capture *capture;
    // For our kitti format data
    std::vector<std::string> binaryFiles;
    int currentFrame = 0;
};
};

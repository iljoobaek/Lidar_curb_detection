#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>

#include <exception>
#include <experimental/filesystem>
#include <glob.h>
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
        BINARIES_KITTI,
        ROSBAG
    };

public:
    int globStartFrame;
    int globEndFrame;
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
    LidarDataReader(std::string rootPath, std::string fn, int start, int end, bool isDownSample)
        : rootPath(rootPath), filename(fn), isDownSample(isDownSample)
    {
        // type = DataType::BINARIES;
        // for (int i = start; i < end; i++)
        // {
        //     binaryFiles.push_back(getBinaryFile(i, fn));
        // }
        if (rootPath.find("kitti") != std::string::npos)
        {
            type = DataType::BINARIES_KITTI;
        }
        else
        {
            type = DataType::BINARIES;
        }

        std::vector<std::string> xyz = glob_new(rootPath.c_str() + std::string(filename) + std::string("velodyne_points/data/*"));

        int start_count = 0;
        for (auto name : xyz)
        {
            // std::cout << "String :" << name << "\n";
            std::string root_path = rootPath.c_str() + std::string(filename) + std::string("velodyne_points/data/");
            // std::cout << "root_path :" << root_path << "\n";

            std::string substring = std::string(name.begin() + root_path.size(), name.end() - 4);

            int frame_num = std::atoi(substring.c_str());
            binaryFiles.push_back(getBinaryFile(frame_num, fn));
            if (start_count == 0)
            {
                globStartFrame = frame_num;
            }
            start_count = 1;
            globEndFrame = frame_num;
            // binaryFiles.push_back(getBinaryFile(frmae_num, fn));
        }
    }
    ~LidarDataReader() {}

private:
    std::vector<std::string> glob_new(const std::string &pattern)
    {
        glob_t globbuf;
        int err = glob(pattern.c_str(), 0, NULL, &globbuf);
        std::vector<std::string> filenames;
        if (err == 0)
        {
            for (size_t i = 0; i < globbuf.gl_pathc; i++)
            {
                filenames.push_back(globbuf.gl_pathv[i]);
            }

            globfree(&globbuf);
            return filenames;
        }
        else
        {
            filenames.push_back("0");
            return filenames;
        }
    }

    void laser_to_cartesian(std::vector<velodyne::Laser> &lasers, std::vector<std::vector<float>> &pointcloud, float theta, cv::Mat &rot, cv::Mat &trans)
    {
        pointcloud.clear();
        pointcloud.reserve(lasers.size());
        for (int i = 0; i < lasers.size(); i++)
        {
            const double distance = static_cast<double>(lasers[i].distance);
            const double azimuth = lasers[i].azimuth * CV_PI / 180.0;
            const double vertical = lasers[i].vertical * CV_PI / 180.0;
            float x = static_cast<float>((distance * std::cos(vertical)) * std::sin(azimuth));
            float y = static_cast<float>((distance * std::cos(vertical)) * std::cos(azimuth));
            float z = static_cast<float>((distance * std::sin(vertical)));

            if (x == 0.0f && y == 0.0f && z == 0.0f)
                continue;

            x /= 100.0, y /= 100.0, z /= 100.0;
            float intensity = static_cast<float>(lasers[i].intensity);
            float ring = static_cast<float>(lasers[i].id);
            float dist = std::sqrt(x * x + y * y + z * z);
            if (dist < 0.9f)
                continue;
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
        //std::cout << "Read in " << pointcloud.size() << " points\n";
        return pointcloud;
    }
    int getScanNumber(float x, float y, float z, std::vector<double> &bounds)
    {
        double angle = std::atan(z / std::sqrt(x * x + y * y));
        int scanNumber = 0;
        while (angle > bounds[scanNumber])
        {
            scanNumber++;
            if (scanNumber == 63)
                break;
        }
        return scanNumber;
    }
    std::vector<std::vector<float>> readFromBinaryKitti(std::string &filename)
    {
        // std::cout << "Filename:" << filename << "\n";
        std::vector<double> centers;

        // This is a file that we need tp ha(center.out)
        std::ifstream in("centers.out");

        std::string line;
        if (in.is_open())
        {
            while (std::getline(in, line))
            {
                double angle = std::stod(line);
                centers.push_back(angle);
            }
        }
        in.close();
        std::reverse(centers.begin(), centers.end());
        std::vector<double> bounds;
        for (int i = 0; i < centers.size() - 1; i++)
        {
            bounds.push_back((centers[i] + centers[i + 1]) / 2);
        }
        std::set<int> downSampleNumbers({3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47, 51, 55, 59, 63});
        std::vector<std::vector<float>> pointcloud;
        int32_t num = 1000000;
        float *data = (float *)malloc(num * sizeof(float));
        float *px = data, *py = data + 1, *pz = data + 2, *pi = data + 3;
        FILE *stream = fopen(filename.c_str(), "rb");
        num = fread(data, sizeof(float), num, stream) / 4;
        for (int32_t i = 0; i < num; i++)
        {
            float dist = std::sqrt((*px) * (*px) + (*py) * (*py) + (*pz) * (*pz));
            float theta = std::atan2(*py, *px) * 180.0f / CV_PI;
            // Find the ring number here
            int ring = getScanNumber(*px, *py, *pz, bounds);
            if (ring >= 0 && ring < 64)
            {
                if (isDownSample)
                {
                    if (downSampleNumbers.find(ring) != downSampleNumbers.end())
                    {
                        ring /= 4;
                        if (dist > 0.9f && *px >= 0.0f)
                            pointcloud.push_back({*px, *py, *pz, *pi, (float)ring, dist, theta});
                    }
                }
                else
                {
                    if (dist > 0.9f && *px >= 0.0f)
                        pointcloud.push_back({*px, *py, *pz, *pi, (float)ring, dist, theta});
                }
            }
            px += 4, py += 4, pz += 4, pi += 4;
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
    std::string getBinaryFile(int frame_idx, std::string root_dir = "20190828124709/")
    {
        std::stringstream ss;
        ss << std::setfill('0') << std::setw(10) << frame_idx;
        // std::string filename = "/home/rtml/lidar_radar_fusion_curb_detection/data/" + root_dir + "velodyne_points/data/" + ss.str() + ".bin";
        std::string filename = rootPath + root_dir + "velodyne_points/data/" + ss.str() + ".bin";
        return filename;
    }
    bool isRun()
    {
        if (type == DataType::BINARIES || type == DataType::BINARIES_KITTI)
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
    void operator>>(std::vector<std::vector<float>> &pointcloud)
    {
        if (type == DataType::BINARIES)
        {
            pointcloud = readFromBinary(binaryFiles[currentFrame++]);
        }
        else if (type == DataType::BINARIES_KITTI)
        {
            pointcloud = readFromBinaryKitti(binaryFiles[currentFrame++]);
        }
        else if (type == DataType::PCAP)
        {
            std::vector<velodyne::Laser> lasers;
            *capture >> lasers;
            laser_to_cartesian(lasers, pointcloud, theta, rot, trans);
        }
    }

private:
    std::string rootPath;
    std::string filename;
    // Sensor info
    float theta;
    cv::Mat rot, trans;
    DataType type;
    bool isDownSample = true;

    // For pcap file by VelodyneCapture
    velodyne::VLP16Capture *capture;
    // For our kitti format data
    std::vector<std::string> binaryFiles;
    int currentFrame = 0;
};
}; // namespace DataReader

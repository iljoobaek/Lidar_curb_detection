#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <experimental/filesystem>
#include <memory>

// Include VelodyneCapture Header
#include "VelodyneCapture.h"
namespace fs = std::experimental::filesystem;

namespace DataSource
{
void laser_to_cartesian(std::vector<velodyne::Laser> &lasers, std::vector<std::vector<float>> &pointcloud, float theta, cv::Mat &rot, cv::Mat &trans) 
{
    pointcloud.clear();
    pointcloud.resize(lasers.size());
    int idx = 0;
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
        pointcloud[idx] = {x, y, z, intensity, ring, dist, azimuth_rot}; // Write to pointcloud
        idx++;
    }
    pointcloud.resize(idx);
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
        float theta = std::atan2(*py, *px) * 180.0f / PI;
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
    float theta = tilted_angle * PI / 180.0f;
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
        float xx = point[0] * std::cos(PI / 2) + point[1] * (-std::sin(PI / 2));
        float yy = point[0] * std::sin(PI / 2) + point[1] * std::cos(PI / 2);
        point[0] = xx;
        point[1] = yy;
    }
}
}

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
            pointcloud = DataSource::readFromBinary(binaryFiles[currentFrame++]);
        }
        else if (type == DataType::PCAP)
        {
            std::vector<velodyne::Laser> lasers;
            *capture >> lasers;
            DataSource::laser_to_cartesian(lasers, pointcloud, theta, rot, trans);
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

namespace LidarViewer 
{
void cvViz3dCallbackSetting(cv::viz::Viz3d &viewer, bool &pause)
{
    // Register Keyboard Callback
    viewer.registerKeyboardCallback(
        []( const cv::viz::KeyboardEvent& event, void* cookie )
        {
        // Close Viewer
        if( event.code == 'q' && event.action == cv::viz::KeyboardEvent::Action::KEY_DOWN )
        {
            static_cast<cv::viz::Viz3d*>( cookie )->close();
        }
        }
        , &viewer);
    viewer.registerKeyboardCallback(
        []( const cv::viz::KeyboardEvent& event, void* pause )
        {
        // Switch state of pause / resume when pressing p
        if( event.code == 'p' && event.action == cv::viz::KeyboardEvent::Action::KEY_DOWN )
        {
            bool* p = static_cast<bool*>( pause );
            *p = !(*p);
        }
        }
        , &pause);
}

void updateViewerFromBuffers(std::vector<std::vector<cv::Vec3f>> &buffers, std::vector<std::vector<bool>> &results, cv::viz::Viz3d &viewer, std::vector<std::vector<float>> &vscanRes, std::vector<cv::viz::WPolyLine> &polyLine) 
{
    // if (buffers[0].empty()) {return;}
    viewer.removeAllWidgets();
    cv::viz::WCloudCollection collection;
    std::vector<cv::Vec3f> curbsBuffer;
    for (int i = 0; i < buffers.size(); i++) 
    {
        int idx = 0;
        for (int j = 0; j < buffers[i].size(); j++) 
        {
            if (results[i][j]) 
            {
                curbsBuffer.push_back(buffers[i][j]);
            } else 
            {
                buffers[i][idx++] = buffers[i][j];
            }
        }
        buffers[i].resize(idx);
    }
    buffers.push_back(curbsBuffer); 

    std::vector<cv::viz::WLine> lines;
    int cnt = 0; 
    for (auto &res : vscanRes)
    {
        cv::viz::WLine line(cv::Point3f(res[0], res[1], res[2]), cv::Point3f(res[0], res[1], res[3]), cv::viz::Color::green());
        line.setRenderingProperty(cv::viz::LINE_WIDTH, 5.0); 
        viewer.showWidget("Line Widget"+std::to_string(cnt++), line);
    }
    
    for (int i = 0; i < buffers.size(); i++) 
    {
        cv::Mat cloudMat = cv::Mat(static_cast<int>(buffers[i].size()), 1, CV_32FC3, &buffers[i][0]);
        if (i == buffers.size()-1) 
        {
            collection.addCloud(cloudMat, cv::viz::Color::red());
        }
        else 
        {
            collection.addCloud(cloudMat, cv::viz::Color::white());
        }
    }
    viewer.showWidget("Poly Left", polyLine[0]);
    viewer.showWidget("Poly Right", polyLine[1]);
    viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
    viewer.showWidget("Cloud", collection);
    viewer.setRenderingProperty("Cloud", cv::viz::POINT_SIZE, 2.0);  // Set point size of the point cloud
    viewer.setBackgroundColor(cv::viz::Color::mlab()); 
    viewer.spinOnce();
}

void updateViewerFromBuffers(std::vector<std::vector<cv::Vec3f>> &buffers, std::vector<std::vector<int>> &results, cv::viz::Viz3d &viewer, std::vector<std::vector<float>> &vscanRes) 
{
    // if (buffers[0].empty()) {return;}
    cv::viz::WCloudCollection collection;

    std::vector<cv::Vec3f> curbsBuffer_1;
    std::vector<cv::Vec3f> curbsBuffer_2;
    for (int i = 0; i < buffers.size(); i++) 
    {
        int idx = 0;
        for (int j = 0; j < buffers[i].size(); j++) 
        {
            if (results[i][j] == 1) 
            {
                curbsBuffer_1.push_back(buffers[i][j]);
            } 
            else if (results[i][j] == -1) 
            {
                curbsBuffer_2.push_back(buffers[i][j]);
            } 
            else 
            {
                buffers[i][idx++] = buffers[i][j];
            }
        }
        buffers[i].resize(idx);
    }
    buffers.push_back(curbsBuffer_1); 
    buffers.push_back(curbsBuffer_2); 
    
    for (auto &res : vscanRes)
    {
        std::cout << res[0] << " " << res[1] << " " << res[2] << " " << res[3] << std::endl;
        cv::viz::WLine line(cv::Point3f(res[0], res[1], res[2]), cv::Point3f(res[0], res[1], res[3]), cv::viz::Color::green());
        line.setRenderingProperty(cv::viz::LINE_WIDTH, 5.0); 
        viewer.showWidget("Line Widget", line);
    }
    
    for (int i = 0; i < buffers.size(); i++) 
    {
        cv::Mat cloudMat = cv::Mat(static_cast<int>(buffers[i].size()), 1, CV_32FC3, &buffers[i][0]);
        if (i == buffers.size()-1) 
        {
            collection.addCloud(cloudMat, cv::viz::Color::red());
        }
        else if (i == buffers.size()-2) 
        {
            collection.addCloud(cloudMat, cv::viz::Color::green());
        }
        else 
        {
            collection.addCloud(cloudMat, cv::viz::Color::white());
        }
    }

    viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
    viewer.showWidget("Cloud", collection);
    viewer.setRenderingProperty("Cloud", cv::viz::POINT_SIZE, 2.0);  // Set point size of the point cloud
    viewer.setBackgroundColor(cv::viz::Color::mlab()); 
    viewer.spinOnce();
}

void updateViewerFromSingleBuffer(std::vector<cv::Vec3f> &buffer, cv::viz::Viz3d &viewer) 
{
    cv::Mat cloudMat = cv::Mat(static_cast<int>(buffer.size()), 1, CV_32FC3, &buffer[0]);
    cv::viz::WCloud cloud( cloudMat );
    viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
    viewer.showWidget("Cloud", cloud);
    viewer.spinOnce();
}

void pushToBuffer(std::vector<cv::Vec3f> &buffer, const std::vector<std::vector<float>> &pointcloud) 
{
    buffer.resize(pointcloud.size());
    for (int i = 0; i < pointcloud.size(); i++) 
    {
        buffer[i] = cv::Vec3f( pointcloud[i][0], pointcloud[i][1], pointcloud[i][2] ); 
    }
}

void push_result_to_buffer(std::vector<cv::Vec3f> &buffer, const std::vector<std::vector<float>> &pointcloud, cv::Mat &rot, cv::Mat &trans) 
{
    buffer.resize(pointcloud.size());
    for (int i = 0; i < pointcloud.size(); i++) 
    {
        buffer[i] = cv::Vec3f( pointcloud[i][0], pointcloud[i][1], pointcloud[i][2] ); 
        // cv::Mat p(buffer[i]);
        // float temp_x = p.at<float>(0,0), temp_y = p.at<float>(1,0); 
        // buffer[i][0] = temp_y + trans.at<float>(0,0);
        // buffer[i][1] = temp_x + trans.at<float>(1,0);
        // buffer[i][2] = -buffer[i][2] + trans.at<float>(2,0);
    }
}
}
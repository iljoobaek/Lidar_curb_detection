#include <iostream>
#include <vector>
#include <string> // memset()
#include <set>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <stdexcept>
#include <sstream>
#include <glob.h> // glob(), globfree()
#include <experimental/filesystem>

// Include VelodyneCapture Header
#include "VelodyneCapture.h"
namespace fs = std::experimental::filesystem;



// Usage: std::vector<std::string> xyz = glob("/some/path/img*.png")
// std::vector<std::string> glob_new(const std::string& pattern) {
//     using namespace std;

//     // glob struct resides on the stack
//     glob_t glob_result;
//     memset(&glob_result, 0, sizeof(glob_result));

//     // do the glob operation
//     int return_value = glob(pattern.c_str(), GLOB_TILDE, NULL, &glob_result);
//     if(return_value != 0) {
//         globfree(&glob_result);
//         stringstream ss;
//         ss << "glob() failed with return_value " << return_value << endl;
//         throw std::runtime_error(ss.str());
//     }

//     // collect all the filenames into a std::list<std::string>
//     vector<string> filenames;
//     for(size_t i = 0; i < glob_result.gl_pathc; ++i) {
//         filenames.push_back(string(glob_result.gl_pathv[i]));
//     }

//     // cleanup
//     globfree(&glob_result);

//     // done
//     return filenames;
// }




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
            std::cout<<fn.c_str()<<" \n";
            capture = new velodyne::VLP16Capture(fn); 
            std::cout<<"Made captures of vlp \n";
        }
    }

    LidarDataReader(std::string fn,float theta, cv::Mat &rot, cv::Mat &trans) 
    {
        if (fn.find(".bag") != std::string::npos)
        {
            type = DataType::ROSBAG; 
        }
        else if (fn.find(".pcap") != std::string::npos)
        {
            this->theta = theta;
            this->rot = rot;
            this->trans = trans;
            type = DataType::PCAP;
            std::cout<<fn.c_str()<<" \n";
            capture = new velodyne::VLP16Capture(fn); 
            std::cout<<"Made captures of vlp \n";
        }
    }

    LidarDataReader(std::string rootPath, std::string fn, int start, int end, bool isDownSample) 
                                    : rootPath(rootPath), filename(fn), isDownSample(isDownSample)
    {
        if (rootPath.find("kitti") != std::string::npos)
        {
            type = DataType::BINARIES_KITTI; 
        }
        else
        {
            type = DataType::BINARIES;
        }
        std::vector<std::string> xyz = glob_new( rootPath.c_str() + std::string(filename) + std::string("velodyne_points/data/*"));
        int start_count = 0;
        for(auto name : xyz){
            std::string root_path = rootPath.c_str() + std::string(filename) + std::string("velodyne_points/data/");
            std::string substring =  std::string(name.begin() + root_path.size(), name.end()-4) ;
            
            int frame_num = std::atoi(substring.c_str());
            binaryFiles.push_back(getBinaryFile(frame_num, fn));
            if(start_count == 0){
                globStartFrame = frame_num;
            }
            start_count = 1;
            globEndFrame =frame_num;
            // binaryFiles.push_back(getBinaryFile(frmae_num, fn));
        }
        // for (int i = start; i < end; i++)
        // {
        //     binaryFiles.push_back(getBinaryFile(i, fn));
        // } 
    }
    ~LidarDataReader() {}
private:

    void push_result_to_buffer_vector_viewer(std::vector<cv::Vec3f> &buffer, const std::vector<std::vector<float>> &pointcloud, cv::Mat &rot, cv::Mat &trans) 
    {
        buffer.resize(pointcloud.size());
        for (int i = 0; i < pointcloud.size(); i++) 
        {
            std::vector<float> stating;
            buffer[i] = cv::Vec3f( pointcloud[i][0], pointcloud[i][1], pointcloud[i][2] ); 
            cv::Mat p(buffer[i]);
            p = rot * p; // Rotation
            float temp_x = p.at<float>(0,0);
            float temp_y = p.at<float>(1,0);
            float temp_z = p.at<float>(2,0);
            buffer[i][0] = temp_y + trans.at<float>(0,0);
            buffer[i][1] = temp_x + trans.at<float>(1,0);
            //buffer[i][2] = -buffer[i][2] + trans.at<float>(2,0);
            buffer[i][2] = -temp_z + trans.at<float>(2,0);

            float tmp_x = buffer[i][0];
            buffer[i][0] = buffer[i][1];
            buffer[i][1] = tmp_x;
            buffer[i][2] = -buffer[i][2];   
        }
        
        for(int i=0; i<buffer.size();i++){
            float *p = (float*)pointcloud[i].data();
            p[0] = buffer[i][0];
            p[1] = buffer[i][1];
            p[2] = buffer[i][2]; 
        }
    }

    void laser_to_cartesian_viewer(std::vector<velodyne::Laser> &lasers, std::vector<std::vector<float>> &pointcloud, float theta, cv::Mat &rot, cv::Mat &trans) {
        pointcloud.clear();
        pointcloud.resize(lasers.size());
        int idx = 0;
        for (int i = 0; i < lasers.size(); i++) {
            //double azimuth_rot = lasers[i].azimuth + theta;
            double azimuth_rot = lasers[i].azimuth;
            if (azimuth_rot >= 360.0) {
                azimuth_rot -= 360.0;
            }
            else if (azimuth_rot < 0.0) {
                azimuth_rot += 360.0;
            }
            const double distance = static_cast<double>( lasers[i].distance );
            const double azimuth  = azimuth_rot  * CV_PI / 180.0;
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
            
            pointcloud[idx] = {x, y, z, intensity, ring, dist, static_cast<float>(azimuth_rot)}; // Write to pointcloud
            idx++;
        }
        pointcloud.resize(idx);
    }



    void laser_to_cartesian(std::vector<velodyne::Laser> &lasers, std::vector<std::vector<float>> &pointcloud, float theta, cv::Mat &rot, cv::Mat &trans) 
    {
        pointcloud.clear();
        pointcloud.reserve(lasers.size());
        for (int i = 0; i < lasers.size(); i++) 
        {
            const double distance = static_cast<double>( lasers[i].distance );
            const double azimuth  = 180 - lasers[i].azimuth  * CV_PI / 180.0;
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
    
    std::vector<std::string> glob_new(const std::string& pattern)  {
        glob_t globbuf;
        int err = glob(pattern.c_str(), 0, NULL, &globbuf);
        std::vector<std::string> filenames;
        if(err == 0)
        {
            for (size_t i = 0; i < globbuf.gl_pathc; i++)
            {
                filenames.push_back(globbuf.gl_pathv[i]);
            }

            globfree(&globbuf);
            return filenames;
        }
        else{
            filenames.push_back("0");
            return filenames;
        }
    }


    std::vector<std::vector<float>> readFromBinaryKitti(std::string &filename, std::string area)
    {
        std::vector<double> centers;
        
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
        for (int i = 0; i < centers.size()-1; i++)
        {
            bounds.push_back((centers[i] + centers[i+1]) / 2);
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
            int ring = getScanNumber(-*px, -*py, *pz , bounds);
            if (ring >= 0 && ring < 64)
            {
                if (isDownSample)
                {
                    std::cout<<"Downsampling ... \n";
                    if (downSampleNumbers.find(ring) != downSampleNumbers.end()) 
                    {
                        ring /= 4;
                        if (dist > 0.9f && *px >= 0.0f)
                            pointcloud.push_back({*px, *py, *pz, *pi, (float)ring, dist, theta});
                    }
                }
                else
                {
                    if( area.compare("front") == 0 ){
                       if (dist > 0.9f && *px >= 0.0f)
                        pointcloud.push_back({*px, *py, *pz, *pi, (float)ring, dist, theta});
                    }
                    else if( area.compare("back") == 0 ){
                       if (dist > 0.9f && *px <= 0.0f)
                        pointcloud.push_back({*px, *py, *pz, *pi, (float)ring, dist, theta});
                    }
                    else{
                        std::cout<<"Wrong area selection \n";
                    }
                }
            }
            px += 4, py += 4, pz += 4, pi += 4;
        }
        fclose(stream);
        std::cout << "Read in " << pointcloud.size() << " points\n";
        return pointcloud;
    }
    int getScanNumber(float x, float y, float z, std::vector<double> &bounds)
    {
        double angle = std::atan(z / std::sqrt(x * x + y * y));
        int scanNumber = 0;
        while (angle > bounds[scanNumber])
        {
            scanNumber++;
            if (scanNumber == 63) break;
        }
        return scanNumber;
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
    
    
    void operator >>(std::vector<std::vector<float>> &pointcloud) 
    {
        if (type == DataType::BINARIES)
        {
            pointcloud = readFromBinary(binaryFiles[currentFrame++]);
        }
        else if (type == DataType::BINARIES_KITTI)
        {
            pointcloud = readFromBinaryKitti(binaryFiles[currentFrame++], "front");
        }
        else if (type == DataType::PCAP)
        {
            std::vector<velodyne::Laser> lasers;
            (*capture) >> lasers;
            std::cout<<"Applying laser to cart \n";
            // laser_to_cartesian(lasers, pointcloud, theta, rot, trans);
            // laser_to_cartesian(lasers, pointcloud, this->theta, this->rot, this->trans);
            // std::cout<<"Shape is "<<pointcloud.size()<<" \n";
            // std::vector<cv::Vec3f> buffer;
            // rotate_pointcloud(buffer, pointcloud, this->rot, this->trans);

            laser_to_cartesian_viewer(lasers, pointcloud,  this->theta,  this->rot,  this->trans);
            // Push result to buffer 
            std::cout<<"[DR] Here after l-c\n";
            std::vector<cv::Vec3f> buffer;
            push_result_to_buffer_vector_viewer(buffer, pointcloud,  this->rot,  this->trans);
            std::cout<<"[DR] Theta is "<<this->theta<<"\n";
        }
    }
    
    void operator <<(std::vector<std::vector<float>> &pointcloud) 
    {
        if (type == DataType::BINARIES)
        {
            pointcloud = readFromBinary(binaryFiles[currentFrame++]);
        }
        else if (type == DataType::BINARIES_KITTI)
        {
            pointcloud = readFromBinaryKitti(binaryFiles[currentFrame++], "back");
        }
        else if (type == DataType::PCAP)
        {
            std::vector<velodyne::Laser> lasers;
            *capture >> lasers;
            laser_to_cartesian(lasers, pointcloud, this->theta, this->rot, this->trans);
        }
    }

private:
    std::string rootPath;
    std::string filename;
    // Sensor info
    bool isDownSample = false;
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
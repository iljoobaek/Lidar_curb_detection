#include "boundary_detection.h"

// void update_viewer(std::vector<std::vector<cv::Vec3f>> &buffers, std::vector<cv::viz::WLine> &lines, std::vector<cv::viz::WText3D> &confidences, std::vector<cv::Vec3f> &radar_pointcloud, std::vector<cv::viz::WPolyLine> &thirdOrder, cv::viz::Viz3d &viewer) {
//     if (buffers[0].empty()) {return;}
//     // Create Widget
//     cv::Mat cloudMat = cv::Mat(static_cast<int>(buffers[0].size()), 1, CV_32FC3, &buffers[0][0]);
//     cv::Mat lineMat = cv::Mat(static_cast<int>(buffers[1].size()), 1, CV_32FC3, &buffers[1][0]);

//     cv::Mat radarMat = cv::Mat(static_cast<int>(radar_pointcloud.size()),
//         1, CV_32FC3, &radar_pointcloud[0]);

//     cv::viz::WCloudCollection collection;
//     collection.addCloud(radarMat, cv::viz::Color::yellow());
//     collection.addCloud(cloudMat, cv::viz::Color::white());
//     collection.addCloud(lineMat, cv::viz::Color::red());
//     // Show Point Cloudcloud

//     viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
//     viewer.showWidget("Cloud", collection);
//     if (lines.size()) {
//         viewer.showWidget("LidarLine Left", lines[0]);
//         viewer.showWidget("LidarLine Right", lines[1]);
//     }
//     viewer.showWidget("Poly Left", thirdOrder[0]);
//     viewer.showWidget("Poly Right", thirdOrder[1]);
//     viewer.showWidget("Confidence Left", confidences[0]);
//     viewer.showWidget("Confidence Right", confidences[1]);
//     viewer.spinOnce();
// }

// vector<vector<float>> Boundary_detection::read_bin(string filename)
// {
//     vector<vector<float>> pointcloud;
//     int32_t num = 1000000;
//     float *data = (float *)malloc(num * sizeof(float));
//     float *px = data, *py = data + 1, *pz = data + 2, *pi = data + 3, *pr = data + 4;

//     FILE *stream = fopen(filename.c_str(), "rb");
//     num = fread(data, sizeof(float), num, stream) / 5;
//     for (int32_t i = 0; i < num; i++)
//     {
//         float dist = std::sqrt((*px) * (*px) + (*py) * (*py) + (*pz) * (*pz));
//         float theta = std::atan2(*py, *px) * 180.0f / PI;
//         if (dist > 0.9f && *px >= 0.0f)
//             pointcloud.push_back({*px, *py, *pz, *pi, *pr, dist, theta});
//         px += 5, py += 5, pz += 5, pi += 5, pr += 5;
//     }
//     fclose(stream);
//     this->pointcloud_unrotated = vector<vector<float>>(pointcloud.begin(), pointcloud.end());
//     cout << "Read in " << pointcloud.size() << " points\n";
//     return pointcloud;
// }

void Boundary_detection::rotate_and_translate_multi_lidar_yaw(const cv::Mat &rot)
{
    if (this->pointcloud.empty()) return;
    
    for (auto &point : this->pointcloud)
    {
        float x = point[0] * rot.at<float>(0,0) + point[1] * rot.at<float>(0,1);
        float y = point[0] * rot.at<float>(1,0) + point[1] * rot.at<float>(1,1);
        point[0] = x;
        point[1] = y;
    }
}

void Boundary_detection::max_height_filter(float max_height)
{
    int cur = 0;
    for (int i = 0; i < this->pointcloud.size(); i++)
    { 
        if (this->pointcloud[i][2] < max_height)
        {
            this->pointcloud[cur] = this->pointcloud[i];
            cur++;
        }
    }
    this->pointcloud.erase(this->pointcloud.begin() + cur, this->pointcloud.end());
}

void Boundary_detection::rearrange_pointcloud() 
{
    std::vector<std::vector<float>> pointcloud_copy(this->pointcloud.begin(), this->pointcloud.end());
    int cur_idx = 0;
    for (int i = 0; i < num_of_scan; i++)
    {
        this->ranges[i * 2][0] = cur_idx;
        auto iter = pointcloud_copy.begin();
        while ((iter = std::find_if(iter, pointcloud_copy.end(), [&](const std::vector<float> &point) { return point[4] == static_cast<float>(i) && point[6] > 0; })) != pointcloud_copy.end())
        {
            this->pointcloud[cur_idx++] = (*iter);
            iter++;
        }
        this->ranges[i * 2][1] = cur_idx;
        this->ranges[i * 2 + 1][0] = cur_idx;
        iter = pointcloud_copy.begin();
        while ((iter = std::find_if(iter, pointcloud_copy.end(), [&](const std::vector<float> &point) { return point[4] == static_cast<float>(i) && point[6] <= 0; })) != pointcloud_copy.end())
        {
            this->pointcloud[cur_idx++] = (*iter);
            iter++;
        }
        this->ranges[i * 2 + 1][1] = cur_idx;
    }
    assert(cur_idx == this->pointcloud.size());
}

void Boundary_detection::pointcloud_preprocessing(const cv::Mat &rot)
{
    rotate_and_translate_multi_lidar_yaw(rot);
    reset();
    // auto t_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    // auto t_end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t_start;
}

void Boundary_detection::detect(const cv::Mat &rot, const cv::Mat &trans, bool vis) 
{
    pointcloud_preprocessing(rot);
    is_boundary = std::vector<bool>(pointcloud.size(), false);
}

void Boundary_detection::reset()
{
}

std::vector<std::vector<float>>& Boundary_detection::get_pointcloud() 
{
    return pointcloud; 
}

std::vector<int>& Boundary_detection::get_result() 
{
    return is_boundary_int;
}

std::vector<bool>& Boundary_detection::get_result_bool() 
{
    return is_boundary;
}

// Show Point Cloud
std::vector<std::vector<cv::Vec3f>> Boundary_detection::getLidarBuffers(const std::vector<std::vector<float>> &pointcloud, const std::vector<bool> &result) 
{
    std::vector<cv::Vec3f> buffer(pointcloud.size());
    std::vector<cv::Vec3f> lineBuffer;
    for (int i = 0; i < pointcloud.size(); i++) 
    {
        if (result[i]) 
        {
            lineBuffer.push_back(cv::Vec3f(pointcloud[i][0], pointcloud[i][1], pointcloud[i][2]));
        } 
        else 
        {
            buffer[i] = cv::Vec3f(pointcloud[i][0], pointcloud[i][1], pointcloud[i][2]);
        }
    }
    std::vector<std::vector<cv::Vec3f>> buffers;
    buffers.push_back(buffer);
    buffers.push_back(lineBuffer);
    return buffers;
}

void Boundary_detection::timedFunction(std::function<void(void)> func, unsigned int interval) 
{
    std::thread([func, interval]() 
    {
        while (true) 
        {
            auto x = std::chrono::steady_clock::now() + std::chrono::milliseconds(interval);
            func();
            std::this_thread::sleep_until(x);
        }
    }).detach();
}

void Boundary_detection::expose() 
{
    this->mem_mutex.lock();
    boost::interprocess::managed_shared_memory segment{boost::interprocess::open_only, "radar_vector"};
    radar_shared *shared = segment.find<radar_shared>("radar_shared").first;
    this->radar_pointcloud.assign(shared->begin(), shared->end());
    this->mem_mutex.unlock();
}

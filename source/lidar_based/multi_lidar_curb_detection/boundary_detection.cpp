#include "boundary_detection.h"

template <typename T>
std::vector<T> conv(std::vector<T> const &f, std::vector<T> const &g)
{
    int const nf = f.size();
    int const ng = g.size();
    int const n = nf + ng - 1;
    std::vector<T> out(n, T());
    for (auto i(0); i < n; ++i)
    {
        int const jmn = (i >= ng - 1) ? i - (ng - 1) : 0;
        int const jmx = (i < nf - 1) ? i : nf - 1;
        for (auto j(jmn); j <= jmx; ++j)
        {
            out[i] += (f[j] * g[i - j]);
        }
    }
    return out;
}

// Show Point Cloud
std::vector<std::vector<cv::Vec3f>> Boundary_detection::getLidarBuffers(const std::vector<std::vector<float>> &pointcloud, const std::vector<bool> &result) {
    std::vector<cv::Vec3f> buffer(pointcloud.size());
    std::vector<cv::Vec3f> lineBuffer;
    for (int i = 0; i < pointcloud.size(); i++) {
        if (result[i]) {
            lineBuffer.push_back(cv::Vec3f(pointcloud[i][0], pointcloud[i][1], pointcloud[i][2]));
        } else {
            buffer[i] = cv::Vec3f(pointcloud[i][0], pointcloud[i][1], pointcloud[i][2]);
        }
    }
    std::vector<std::vector<cv::Vec3f>> buffers;
    buffers.push_back(buffer);
    buffers.push_back(lineBuffer);
    return buffers;
}

// void update_viewer_lidar(const vector<vector<float>> &pointcloud, const vector<bool> &result, std::vector<cv::Point2f> &leftLine, std::vector<cv::Point2f> &rightLine, cv::viz::Viz3d &viewer, bool isPCAP)
// {
//     std::vector<cv::Vec3f> buffer(pointcloud.size());
//     std::vector<cv::Vec3b> colors(pointcloud.size());
//     for (int i = 0; i < pointcloud.size(); i++)
//     {
//         buffer[i] = cv::Vec3f(pointcloud[i][0], pointcloud[i][1], pointcloud[i][2]);
//         if (result[i])
//             colors[i] = {0, 0, 255};
//         else
//             colors[i] = {255, 255, 255};
//     }
//     // Create Widget
//     cv::Mat cloudMat = cv::Mat(static_cast<int>(buffer.size()), 1, CV_32FC3, &buffer[0]);
//     cv::Mat colorMat = cv::Mat(static_cast<int>(colors.size()), 1, CV_8UC3, &colors[0]);
//     cv::viz::WCloud cloud(cloudMat, colorMat);

//     if (!leftLine.empty())
//     {
//         auto points = get_line(leftLine, isPCAP);
//         cv::viz::WLine left(points[0], points[1], cv::viz::Color::green());
//         viewer.showWidget("Line Widget 1", left);
//     }
//     if (!rightLine.empty())
//     {
//         auto points = get_line(rightLine, isPCAP);
//         cv::viz::WLine right(points[0], points[1], cv::viz::Color::green());
//         viewer.showWidget("Line Widget 2", right);
//     }
//     // Show Point Cloudcloud
//     viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
//     viewer.showWidget("Cloud", cloud);
//     viewer.spinOnce();
// }

void update_viewer(std::vector<std::vector<cv::Vec3f>> &buffers, std::vector<cv::viz::WLine> &lines, std::vector<cv::viz::WText3D> &confidences, std::vector<cv::Vec3f> &radar_pointcloud, std::vector<cv::viz::WPolyLine> &thirdOrder, cv::viz::Viz3d &viewer) {
    if (buffers[0].empty()) {return;}
    // Create Widget
    cv::Mat cloudMat = cv::Mat(static_cast<int>(buffers[0].size()), 1, CV_32FC3, &buffers[0][0]);
    cv::Mat lineMat = cv::Mat(static_cast<int>(buffers[1].size()), 1, CV_32FC3, &buffers[1][0]);

    cv::Mat radarMat = cv::Mat(static_cast<int>(radar_pointcloud.size()),
        1, CV_32FC3, &radar_pointcloud[0]);

    cv::viz::WCloudCollection collection;
    collection.addCloud(radarMat, cv::viz::Color::yellow());
    collection.addCloud(cloudMat, cv::viz::Color::white());
    collection.addCloud(lineMat, cv::viz::Color::red());
    // Show Point Cloudcloud

    viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
    viewer.showWidget("Cloud", collection);
    if (lines.size()) {
        viewer.showWidget("LidarLine Left", lines[0]);
        viewer.showWidget("LidarLine Right", lines[1]);
    }
    viewer.showWidget("Poly Left", thirdOrder[0]);
    viewer.showWidget("Poly Right", thirdOrder[1]);
    viewer.showWidget("Confidence Left", confidences[0]);
    viewer.showWidget("Confidence Right", confidences[1]);
    viewer.spinOnce();
}


void Boundary_detection::laser_to_cartesian(std::vector<velodyne::Laser> &lasers) {
    this->pointcloud.clear();
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

        float intensity = static_cast<float>(lasers[i].intensity);
        float ring = static_cast<float>(lasers[i].id);

        x /= 100.0, y /= 100.0, z /= 100.0;
        float dist = std::sqrt(x * x + y * y + z * z);
        float theta = std::atan2(y, x) * 180.0f / PI;
        if (dist > 0.9f && y >= 0.0f)
        {
            this->pointcloud.push_back({x, y, z, intensity, ring, dist, theta});
        }
    }
}

vector<vector<float>> Boundary_detection::read_bin(string filename)
{
    vector<vector<float>> pointcloud;
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
    this->pointcloud_unrotated = vector<vector<float>>(pointcloud.begin(), pointcloud.end());
    cout << "Read in " << pointcloud.size() << " points\n";
    return pointcloud;
}

void Boundary_detection::rotate_and_translate()
{
    if (this->pointcloud.empty()) return;

    // rotation matrix along x
    // [1,           0,           0]
    // [0,  cos(theta), -sin(theta)]
    // [0,  sin(theta),  cos(theta)]

    // rotation matrix along y
    // [cos(theta),   0, sin(theta)]
    // [0,            1,          0]
    // [-sin(theta),  0, cos(theta)]
    
    float theta = this->tilted_angle * PI / 180.0f;
    // cout << "[ "<< std::cos(theta) << " " << 0.0f << " " << std::sin(theta) << "\n";
    // cout << 0.0f << " " << 1.0f << " " << 0.0f << "\n";
    // cout << -std::sin(theta) << " " << 0.0f << " " << std::cos(theta) << " ]"<< "\n";
    if (this->isPCAP)
    {
        theta = -theta;
        for (auto &point : this->pointcloud)
        {
            float y = point[1] * std::cos(theta) + point[2] * (-std::sin(theta));
            float z = point[1] * std::sin(theta) + point[2] * std::cos(theta) + this->sensor_height;
            point[1] = y;
            point[2] = z;
        }
    }
    else
    {
        for (auto &point : this->pointcloud)
        {
            float x = point[0] * std::cos(theta) + point[2] * std::sin(theta);
            float z = point[0] * (-std::sin(theta)) + point[2] * std::cos(theta) + this->sensor_height;
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

void Boundary_detection::max_height_filter(float max_height, bool is_ground_extracted)
{
    // if (this->isPCAP) this->pointcloud_unrotated = std::vector<std::vector<float>>(this->pointcloud.begin(), this->pointcloud.end());
    this->pointcloud_unrotated = std::vector<std::vector<float>>(this->pointcloud.begin(), this->pointcloud.end());
    int cur = 0;
    for (int i = 0; i < this->pointcloud.size(); i++)
    { 
        if (this->pointcloud[i][2] < max_height)
        {
            this->pointcloud[cur] = this->pointcloud[i];
            this->pointcloud_unrotated[cur] = this->pointcloud_unrotated[i];
            if (is_ground_extracted) this->is_obstacle[cur] = this->is_obstacle[i];
            cur++;
        }
    }
    this->pointcloud.erase(this->pointcloud.begin() + cur, this->pointcloud.end());
    this->pointcloud_unrotated.erase(this->pointcloud_unrotated.begin() + cur, this->pointcloud_unrotated.end());
    if (is_ground_extracted) this->is_obstacle.erase(this->is_obstacle.begin() + cur, this->is_obstacle.end());
}
#if USE_MULTIPLE_LIDAR
void Boundary_detection::rearrange_pointcloud() 
{
    std::vector<std::vector<float>> pointcloud_copy(this->pointcloud.begin(), this->pointcloud.end());
    int cur_idx = 0;
    for (int i = 0; i < num_of_scan; i++)
    {
        this->ranges[i][0] = cur_idx;
        auto iter = pointcloud_copy.begin();
        while ((iter = std::find_if(iter, pointcloud_copy.end(), [&](const vector<float> &point) { return point[4] == static_cast<float>(i); })) != pointcloud_copy.end())
        {
            this->pointcloud[cur_idx++] = (*iter);
            iter++;
        }
        this->ranges[i][1] = cur_idx;
    }
    assert(cur_idx == this->pointcloud.size());
}

void Boundary_detection::rearrange_pointcloud_unrotated() 
{
    std::vector<std::vector<float>> pointcloud_unrotated_copy(this->pointcloud_unrotated.begin(), this->pointcloud_unrotated.end());
    int cur_idx2 = 0;
    for (int i = 0; i < num_of_scan; i++)
    {
        auto iter2 = pointcloud_unrotated_copy.begin();
        while ((iter2 = std::find_if(iter2, pointcloud_unrotated_copy.end(), [&](const vector<float> &point) { return point[4] == static_cast<float>(i); })) != pointcloud_unrotated_copy.end())
        {
            this->pointcloud_unrotated[cur_idx2++] = (*iter2);
            iter2++;
        }
    }
    assert(cur_idx2 == this->pointcloud_unrotated.size());
}
#else
void Boundary_detection::rearrange_pointcloud() 
{
    std::vector<std::vector<float>> pointcloud_copy(this->pointcloud.begin(), this->pointcloud.end());
    int cur_idx = 0;
    for (int i = 0; i < num_of_scan; i++)
    {
        this->ranges[i * 2][0] = cur_idx;
        auto iter = pointcloud_copy.begin();
        while ((iter = std::find_if(iter, pointcloud_copy.end(), [&](const vector<float> &point) { return point[4] == static_cast<float>(i) && point[6] > 0; })) != pointcloud_copy.end())
        {
            this->pointcloud[cur_idx++] = (*iter);
            iter++;
        }
        this->ranges[i * 2][1] = cur_idx;
        this->ranges[i * 2 + 1][0] = cur_idx;
        iter = pointcloud_copy.begin();
        while ((iter = std::find_if(iter, pointcloud_copy.end(), [&](const vector<float> &point) { return point[4] == static_cast<float>(i) && point[6] <= 0; })) != pointcloud_copy.end())
        {
            this->pointcloud[cur_idx++] = (*iter);
            iter++;
        }
        this->ranges[i * 2 + 1][1] = cur_idx;
    }
    assert(cur_idx == this->pointcloud.size());
}

void Boundary_detection::rearrange_pointcloud_unrotated() 
{
    std::vector<std::vector<float>> pointcloud_unrotated_copy(this->pointcloud_unrotated.begin(), this->pointcloud_unrotated.end());
    int cur_idx2 = 0;
    for (int i = 0; i < num_of_scan; i++)
    {
        auto iter2 = pointcloud_unrotated_copy.begin();
        while ((iter2 = std::find_if(iter2, pointcloud_unrotated_copy.end(), [&](const vector<float> &point) { return point[4] == static_cast<float>(i) && point[6] > 0; })) != pointcloud_unrotated_copy.end())
        {
            this->pointcloud_unrotated[cur_idx2++] = (*iter2);
            iter2++;
        }
        iter2 = pointcloud_unrotated_copy.begin();
        while ((iter2 = std::find_if(iter2, pointcloud_unrotated_copy.end(), [&](const vector<float> &point) { return point[4] == static_cast<float>(i) && point[6] <= 0; })) != pointcloud_unrotated_copy.end())
        {
            this->pointcloud_unrotated[cur_idx2++] = (*iter2);
            iter2++;
        }
    }
    assert(cur_idx2 == this->pointcloud_unrotated.size());
}
#endif

void Boundary_detection::rearrange_pointcloud_sort() {
    std::sort(this->pointcloud.begin(), this->pointcloud.end(), 
        [](const std::vector<float> &p1, const std::vector<float> &p2){
            if (p1[4] == p2[4]) return p1[6] > p2[6];
            else return p1[4] < p2[4]; });
}

std::vector<float> Boundary_detection::get_dist_to_origin() {
    std::vector<float> dist(this->pointcloud.size());
    for (int i = 0; i < dist.size(); i++) 
        dist[i] = std::sqrt(pointcloud[i][0]*pointcloud[i][0] + pointcloud[i][1]*pointcloud[i][1] + pointcloud[i][2]*pointcloud[i][2]); 
    return dist;
}

std::vector<float> Boundary_detection::get_theoretical_dist() {
    std::vector<float> dist(this->num_of_scan);
    for (int i = 0; i < dist.size(); i++) {
        float angle = -(this->angles[i] - this->tilted_angle) * PI / 180.0;
        dist[i] = this->sensor_height / std::sin(angle);
    }
    return dist;
}

void Boundary_detection::object_extraction()
{
    this->is_obstacle = std::vector<bool>(pointcloud.size(), false);
    this->is_boundary = std::vector<bool>(pointcloud.size(), false);
    std::unordered_map<float, std::vector<int>> azimuth_groups; // <azimuth, vector of index>
    std::vector<int> order = {0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7 ,9, 11, 13, 15}; 
    for (int i = 0; i < pointcloud.size(); i++)
    {
        float azimuth = pointcloud[i][6];
        if (azimuth_groups.find(azimuth) == azimuth_groups.end()) azimuth_groups[azimuth] = std::vector<int>(16, -1); 
        azimuth_groups[azimuth][(int)pointcloud[i][4]] = i;
    }
    for (auto &point : azimuth_groups)
    {
        if (point.second[0] == -1)
        {
            continue;
        }
        float azimuth = point.first;
        float min_height = FLT_MAX;
        std::vector<float> dists;
        std::vector<int> scan_idx; // scan lines ordered from -15 to 15 degree, skipped if no laser return
        std::vector<int> scan_order; // scan lines ordered from -15 to 15 degree, skipped if no laser return
        int cur = 0;
        for (int i = 0; i < order.size(); i++)
        {
            if (point.second[order[i]] != -1)
            {
                int idx = point.second[order[i]];
                float x = pointcloud[idx][0], y = pointcloud[idx][1];
                dists.push_back(std::sqrt(x*x + y*y));
                scan_idx.push_back(idx);
                scan_order.push_back(i);
                min_height = std::min(min_height, pointcloud[idx][2]);
            }
        }
        float delta = 0.0f;
        std::stack<int> obstacles;
        bool hit_boundary = false;
        for (int i = 1; i < dists.size(); i++)
        {
            if (dists[i]-dists[i-1] < delta)
            {
                if (scan_order[i] > 6)
                {
                    hit_boundary = true;
                }
                if (!hit_boundary)
                {
                    this->is_boundary[scan_idx[i]] = true;
                    hit_boundary = true;
                }
                obstacles.push(scan_idx[i]); 
            }
            else if (!obstacles.empty()) 
            {
                if (pointcloud[obstacles.top()][2] > min_height + 0.6f)
                {
                    while (!obstacles.empty())
                    {
                        this->is_obstacle[obstacles.top()] = true;
                        this->is_boundary[obstacles.top()] = false;
                        obstacles.pop();
                    }
                }
                else 
                {
                    obstacles = std::stack<int>();
                }
            }
            delta = std::max(dists[i] - dists[i-1], delta);
        }
        if (!obstacles.empty() && pointcloud[obstacles.top()][2] > min_height + 0.6f)
        {
            while (!obstacles.empty())
            {
                this->is_obstacle[obstacles.top()] = true;
                this->is_boundary[obstacles.top()] = false;
                obstacles.pop();
            }
        }
    } 
}

void Boundary_detection::pointcloud_preprocessing()
{
    rotate_and_translate();
    max_height_filter(.45);
    rearrange_pointcloud();
    rearrange_pointcloud_unrotated();
    reset();
}

void Boundary_detection::pointcloud_preprocessing(const cv::Mat &rot)
{
    rotate_and_translate_multi_lidar_yaw(rot);
    max_height_filter(2.0, false);
    rearrange_pointcloud();
    object_extraction();
    // rearrange_pointcloud_unrotated();
    reset();
    // auto t_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    // auto t_end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t_start;
}

float Boundary_detection::dist_between(const std::vector<float> &p1, const std::vector<float> &p2) {
    return std::sqrt((p2[0]-p1[0])*(p2[0]-p1[0]) + (p2[1]-p1[1])*(p2[1]-p1[1]) + (p2[2]-p1[2])*(p2[2]-p1[2]));
}

#if USE_MULTIPLE_LIDAR
std::vector<bool> Boundary_detection::continuous_filter(int scan_id) {
    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
    int n = ed - st;
    std::vector<bool> is_continuous(n, true);
    std::vector<float> thres(this->dist_to_origin.begin()+st, this->dist_to_origin.begin()+ed);
    for (auto &th : thres) {
        th *= THETA_R * 7;
    }
    for (int i = 0; i < n; i++) {
        int j = (i == n-1) ? 0 : i + 1;
        if (dist_between(this->pointcloud[st+i], this->pointcloud[st+j]) > thres[i]) {
            is_continuous[i] = false;
            is_continuous[j] = false;
            this->is_continuous[st + i] = false;
            this->is_continuous[st + j] = false;
        }
    }
    return is_continuous;
}
#else
std::vector<bool> Boundary_detection::continuous_filter(int scan_id) {
    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
    int n = ed - st;
    std::vector<bool> is_continuous(n, true);
    std::vector<float> thres(this->dist_to_origin.begin()+st, this->dist_to_origin.begin()+ed);
    for (auto &t : thres) t *= THETA_R * 7;
    for (int i = 0; i < n-1; i++) {
        if (dist_between(this->pointcloud[st+i], this->pointcloud[st+i+1]) > thres[i]) {
            is_continuous[i] = false;
            is_continuous[i + 1] = false;
            this->is_continuous[st + i] = false;
            this->is_continuous[st + i + 1] = false;
        }
    }
    return is_continuous;
}
#endif

float Boundary_detection::get_angle(const vector<float> &v1, const vector<float> &v2)
{
    float angle;
    float dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    float mag_1 = std::sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
    float mag_2 = std::sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);
    return std::acos(dot_product / (mag_1 * mag_2)) * 180.0 / PI;
}

#if USE_MULTIPLE_LIDAR
std::vector<float> Boundary_detection::direction_change_filter(int scan_id, int k, float angle_thres /* =150.0f */) {
    int n = this->ranges[scan_id][1] - this->ranges[scan_id][0];
    int st = this->ranges[scan_id][0];
    if (scan_id < 8 && scan_id % 2 == 0) 
    {
        k = 16;
    }
    std::vector<float> thres = {120.0f, 140.0f, 120.0f, 140.0f, 120.0f, 140.0f, 120.0f, 140.0f, 
                                120.0f, 150.0f, 125.0f, 150.0f, 125.0f, 150.0f, 125.0f, 150.0f};
    std::vector<float> angles(n, 180.0f);
    if (n - (2 * k) < 0) return angles; 
    
    std::vector<std::vector<float>> direction_vecs_right(n, std::vector<float>(3, 0.0f));
    std::vector<std::vector<float>> direction_vecs_left(n, std::vector<float>(3, 0.0f));

    // might need optimize
    // normalize ?
    for (int i = 0; i < n; i++)
    {
        for (int j = 1; j <= k; j++)
        {
            int idx = (i + j >= n) ? i + j - n : i + j;
            float theta_r = (this->pointcloud[st+idx][6] < this->pointcloud[st+i][6]) ? this->pointcloud[st+idx][6]+360.0f : this->pointcloud[st+idx][6];
            if (std::fabs( theta_r - this->pointcloud[st+i][6]) > 15.0f) continue; 
            direction_vecs_right[i][0] += (this->pointcloud[st + idx][0] - this->pointcloud[st + i][0]);
            direction_vecs_right[i][1] += (this->pointcloud[st + idx][1] - this->pointcloud[st + i][1]);
            direction_vecs_right[i][2] += (this->pointcloud[st + idx][2] - this->pointcloud[st + i][2]);
        }
    }
    for (int i = n - 1; i >= 0; i--)
    {
        for (int j = 1; j <= k; j++)
        {
            int idx = (i - j < 0) ? i - j + n : i - j;
            float theta_l = (this->pointcloud[st+idx][6] > this->pointcloud[st+i][6]) ? this->pointcloud[st+idx][6]-360.0f : this->pointcloud[st+idx][6];
            if (std::fabs( theta_l - this->pointcloud[st+i][6]) > 15.0f) continue; 
            direction_vecs_left[i][0] += (this->pointcloud[st + idx][0] - this->pointcloud[st + i][0]);
            direction_vecs_left[i][1] += (this->pointcloud[st + idx][1] - this->pointcloud[st + i][1]);
            direction_vecs_left[i][2] += (this->pointcloud[st + idx][2] - this->pointcloud[st + i][2]);
        }
    }
    for (int i = 0; i < n; i++)
    {
        angles[i] = get_angle(direction_vecs_left[i], direction_vecs_right[i]);
        if (angles[i] < angle_thres)
        {
            this->is_changing_angle[st + i] = true;
        }
    }
    return angles;
}

std::vector<bool> Boundary_detection::get_local_min(const std::vector<float> &vec) {
    std::vector<int> first_derivative(vec.size()-1);
    for (int i = 0; i < first_derivative.size(); i++) {
        auto diff = vec[i+1] - vec[i];
        if (diff > 0.0f) first_derivative[i] = 1;
        else if (diff < 0.0f) first_derivative[i] = -1;
        else first_derivative[i] = 0;
    }
    std::vector<bool> second_derivative(first_derivative.size()-1, false);
    for (int i = 0; i < second_derivative.size(); i++) {
        auto diff = first_derivative[i+1] - first_derivative[i];
        if (diff > 0) second_derivative[i] = true;
    }
    second_derivative.insert(second_derivative.begin(), false);
    second_derivative.push_back(false);
    return second_derivative;
}

// std::vector<bool> Boundary_detection::local_min_of_direction_change(int scan_id) {
//     int st = this->ranges[scan_id][0];
//     float angle_thres = 150.0f;
//     auto direction = direction_change_filter(scan_id, 8, angle_thres);
//     std::vector<bool> direction_change_local_min(direction.size());
//     std::vector<bool> direction_change(direction.size(), false); 
//     for (int i = 0; i < direction.size(); i++) {
//         if (direction[i] < 150.0f) direction_change[i] = true;
//     }
//     std::vector<bool> local_min = get_local_min(direction);
//     for (int i = 0; i < direction.size(); i++) {
//         direction_change_local_min[i] = direction_change[i] && local_min[i];
//         this->is_local_min[st + i] = direction_change[i] && local_min[i];
//     }
//     return direction_change_local_min;
// }

std::vector<bool> Boundary_detection::local_min_of_direction_change(int scan_id) {
    int st = this->ranges[scan_id][0];
    auto direction = direction_change_filter(scan_id, 5);
    int n = direction.size();
    std::vector<bool> local_min(n, false);
    std::vector<int> confidence(n, 0);
    for (int i = 0; i < n; i++)
    {
        for (int j = -2; j <= 2; j++) 
        {
            int k = i + j;
            if (k < 0) 
            {
                k += n;
            }
            else if (k > n-1) 
            {
                k -= n;
            }
            if (this->is_changing_angle[st + k]) 
            {
                confidence[i]++;
            }
        }
        if (confidence[i] >= 3) 
        {
            this->is_local_min[st+i] = true;
            local_min[i] = true;
        }
    }
    return local_min;
}

#else
std::vector<float> Boundary_detection::direction_change_filter(int scan_id, int k, float angle_thres /* =150.0f */) {
    int n = this->ranges[scan_id][1] - this->ranges[scan_id][0];
    int st = this->ranges[scan_id][0];
    std::vector<float> angles(n, 180.0f);
    if (n - (2 * k) < 0) return angles; 
    
    std::vector<std::vector<float>> direction_vecs_right(n, std::vector<float>(3, 0.0f));
    std::vector<std::vector<float>> direction_vecs_left(n, std::vector<float>(3, 0.0f));

    // might need optimize
    // normalize ?
    for (int i = 0; i < n - k; i++)
    {
        for (int j = i + 1; j <= i + k; j++)
        {
            direction_vecs_right[i][0] = this->pointcloud[st + j][0] - this->pointcloud[st + i][0];
            direction_vecs_right[i][1] = this->pointcloud[st + j][1] - this->pointcloud[st + i][1];
            direction_vecs_right[i][2] = this->pointcloud[st + j][2] - this->pointcloud[st + i][2];
        }
    }
    for (int i = n - 1; i >= k; i--)
    {
        for (int j = i - 1; j >= i - k; j--)
        {
            direction_vecs_left[i][0] = this->pointcloud[st + j][0] - this->pointcloud[st + i][0];
            direction_vecs_left[i][1] = this->pointcloud[st + j][1] - this->pointcloud[st + i][1];
            direction_vecs_left[i][2] = this->pointcloud[st + j][2] - this->pointcloud[st + i][2];
        }
    }
    for (int i = k; i < n - k; i++)
    {
        angles[i] = get_angle(direction_vecs_left[i], direction_vecs_right[i]);
        if (angles[i] < 150.0)
            this->is_changing_angle[st + i] = true;
    }
    return angles;
}

std::vector<bool> get_local_min(const vector<float> &vec) {
    std::vector<int> first_derivative(vec.size()-1);
    for (int i = 0; i < first_derivative.size(); i++) {
        auto diff = vec[i+1] - vec[i];
        if (diff > 0.0f) first_derivative[i] = 1;
        else if (diff < 0.0f) first_derivative[i] = -1;
        else first_derivative[i] = 0;
    }
    std::vector<bool> second_derivative(first_derivative.size()-1, false);
    for (int i = 0; i < second_derivative.size(); i++) {
        auto diff = first_derivative[i+1] - first_derivative[i];
        if (diff > 0) second_derivative[i] = true;
    }
    second_derivative.insert(second_derivative.begin(), false);
    second_derivative.push_back(false);
    return second_derivative;
}

std::vector<bool> Boundary_detection::local_min_of_direction_change(int scan_id) {
    int st = this->ranges[scan_id][0];
    auto direction = direction_change_filter(scan_id, 8);
    std::vector<bool> direction_change_local_min(direction.size());
    std::vector<bool> direction_change(direction.size(), false); 
    for (int i = 0; i < direction.size(); i++) {
        if (direction[i] < 150.0f) direction_change[i] = true;
    }
    std::vector<bool> local_min = get_local_min(direction);
    for (int i = 0; i < direction.size(); i++) {
        direction_change_local_min[i] = direction_change[i] && local_min[i];
        this->is_local_min[st + i] = direction_change[i] && local_min[i];
    }
    return direction_change_local_min;
}
#endif

#if USE_MULTIPLE_LIDAR
std::vector<int> Boundary_detection::elevation_filter_abs(int scan_id) 
{
    // No left or right scan now
    // when azimuth >= 360: ...
    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
    int n = ed - st;
    
    std::vector<int> is_elevation_changing(n, false);
    std::vector<float> z_diff(n, 0.0f); 
    
    std::vector<float> thres_z(n, 0.003);
    // std::vector<float> thres_z(this->dist_to_origin.begin()+st, this->dist_to_origin.begin()+ed);
    // for (auto &th : thres_z) {
    //     th *= THETA_R;
    // }
    for (int i = 0; i < n; i++)
    {
        int j = (i == 0) ? n-1 : i-1;
        z_diff[i] = this->pointcloud[st + i][2] - this->pointcloud[st + j][2];
        if (z_diff[i] > thres_z[i]) 
        {
            is_elevation_changing[i] = 1;
            this->is_elevating[st + i] = 1;
        }
        else if (z_diff[i] < -thres_z[i]) 
        {
            is_elevation_changing[i] = -1;
            this->is_elevating[st + i] = -1;
        }
        else 
        {
            this->is_boundary[st+i] = false;
        }
    }
    return is_elevation_changing;
}

void Boundary_detection::edge_filter_from_elevation(int scan_id, const std::vector<int> &elevation, std::vector<bool> &edge_start, std::vector<bool>& edge_end) 
{
    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
    int n = ed - st;
    int k = 7;
    if (n <= (2 * k)) return; 
    std::vector<int> f_start, f_end;
    // if (scan_id % 2 == 0) 
    // { // left scan
    //     f_start = {-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1};
    //     f_end = {1,1,1,1,1,1,1,1,0,0,0,0,0,0,0};
    // }
    // else
    // {
    //     f_start = {1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1};
    //     f_end = {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
    // }

    auto edge_start_cnt = conv(elevation, f_start);
    auto edge_end_cnt = conv(elevation, f_end);
    for (int i = 0; i < n; i++)
    {
        if (edge_start_cnt[i + k] >= 2)
        {
            edge_start[i] = true;
            this->is_edge_start[st + i] = true;
        }
        if (edge_end_cnt[i + k] >= 6)
        {
            edge_end[i] = true;
            this->is_edge_end[st + i] = true;
        }
    }
}

bool Boundary_detection::is_start_point(int scan_id, int idx, const std::vector<int> &elevation, Boundary_detection::ScanDirection dir) 
{
    int n = elevation.size();
    int k = 7, left_cnt = 0, right_cnt = 0;
    for (int i = 1; i <= k; i++) 
    {
        int j = ((idx+i) >= n) ? idx+i-n : idx+i;
        right_cnt += elevation[j];
    }
    for (int i = 1; i <= k; i++) 
    {
        int j = ((idx-i) < 0) ? idx-i+n : idx-i;
        left_cnt += elevation[j];
    }
    if (dir == Boundary_detection::ScanDirection::CLOCKWISE) 
    {
        if (right_cnt >= 4 && std::abs(left_cnt) < 2) 
        {
            return true;
        }
    }
    else 
    {
        if (left_cnt <= -4 && std::abs(right_cnt) < 2) 
        {
            return true;
        }
    }
    return false;
}

bool Boundary_detection::is_end_point(int scan_id, int idx, const std::vector<int> &elevation, Boundary_detection::ScanDirection dir) 
{
    int n = elevation.size();
    int k = 7, left_cnt = 0, right_cnt = 0;
    for (int i = 1; i <= k; i++) 
    {
        int j = ((idx+i) >= n) ? idx+i-n : idx+i;
        right_cnt += elevation[j];
    }
    for (int i = 1; i <= k; i++) 
    {
        int j = ((idx-i) < 0) ? idx-i+n : idx-i;
        left_cnt += elevation[j];
    }
    if (dir == Boundary_detection::ScanDirection::CLOCKWISE) 
    {
        if (left_cnt >= 4 && std::abs(right_cnt) < 2) 
        {
            return true;
        }
    }
    else 
    {
        if (right_cnt <= -4 && std::abs(left_cnt) < 2) 
        {
            return true;
        }
    }
    return false;
}

#else
std::vector<int> Boundary_detection::elevation_filter(int scan_id) {
    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
    int n = ed - st;
    std::vector<int> is_elevate(n, 0);
    std::vector<float> z_diff(n, 0.0f); 
    float thres_z = 0.005;

    if (scan_id % 2 == 0)
    { // left scan
        for (int i = n - 2; i >= 0; i--)
        {
            z_diff[i] = this->pointcloud[st + i][2] - this->pointcloud[st + i + 1][2];
        }
    }
    else
    {
        for (int i = 1; i < n; i++)
        {
            z_diff[i] = this->pointcloud[st + i][2] - this->pointcloud[st + i - 1][2];
        }
    }
    for (int i = 0; i < n; i++)
    {
        if (this->pointcloud[st + i][6] < 3.0) 
        {
            thres_z = 0.002;
        }
        if (z_diff[i] > thres_z) 
        {
            is_elevate[i] = 1;
        }
    }
    std::vector<int> filter({1,1,1,1,1,1,1,1,1});
    auto res = conv(is_elevate, filter);
    for (int i = 0; i < is_elevate.size(); i++)
    {
        if (res[i + 4] >= 4)
            is_elevate[i] = 1;
        else
            is_elevate[i] = 0;
    }
    for (int i = 0; i < is_elevate.size(); i++)
    {
        if (is_elevate[i] > 0)
            this->is_elevating[st + i] = true;
    }
    return is_elevate;
}

void Boundary_detection::edge_filter_from_elevation(int scan_id, const std::vector<int> &elevation, std::vector<bool> &edge_start, std::vector<bool>& edge_end) 
{
    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
    int n = ed - st;
    int k = 7;
    std::vector<int> f_start, f_end;
    if (n <= (2 * k)) return; 
    if (scan_id % 2 == 0) 
    { // left scan
        f_start = {-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1};
        f_end = {1,1,1,1,1,1,1,1,0,0,0,0,0,0,0};
    }
    else
    {
        f_start = {1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1, -1, -1, -1};
        f_end = {0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1};
    }
    auto edge_start_cnt = conv(elevation, f_start);
    auto edge_end_cnt = conv(elevation, f_end);
    for (int i = 0; i < n; i++)
    {
        if (edge_start_cnt[i + k] >= 2)
        {
            edge_start[i] = true;
            this->is_edge_start[st + i] = true;
        }
        if (edge_end_cnt[i + k] >= 6)
        {
            edge_end[i] = true;
            this->is_edge_end[st + i] = true;
        }
    }
}
#endif

//std::vector<bool> Boundary_detection::obstacle_extraction(int scan_id) 
//{
//    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
//    int n = ed - st;
//    std::vector<bool> is_obstacle(n, false);
//    float height_thres = 0.3, dist = 0.0f;
//    int ring = scan_id / 2;
//    if (scan_id % 2 == 0)
//    { // left scan
//        int i = n - 1, cur_start = -1;
//        while (i >= 0)
//        {
//            if (i == n - 1 && this->pointcloud[st + i][2] > height_thres)
//            {
//                cur_start = i;
//                dist = this->theoretical_dist[ring];
//            }
//            else if (cur_start == -1 && i + 1 < n && !this->is_continuous[i + 1] && !this->is_continuous[i] && this->dist_to_origin[i + 1] > this->dist_to_origin[i])
//            {
//                cur_start = i;
//                dist = this->dist_to_origin[i + 1];
//            }
//            else if (cur_start != -1 && this->dist_to_origin[i] > dist)
//            {
//                for (int j = st + i + 1; j <= st + cur_start; j++)
//                    this->is_obstacle[j] = true;
//                cur_start = -1;
//            }
//            if (i == 0 && cur_start != -1)
//            {
//                for (int j = st + i + 1; j <= st + cur_start; j++)
//                    this->is_obstacle[j] = true;
//            }
//            i--;
//        }
//    }
//    else
//    {
//        int i = 0, cur_start = -1;
//        while (i < n)
//        {
//            if (i == 0 && this->pointcloud[st + i][2] > height_thres)
//            {
//                cur_start = i;
//                dist = this->theoretical_dist[ring];
//            }
//            else if (cur_start == -1 && i - 1 >= 0 && !this->is_continuous[i - 1] && !this->is_continuous[i] && this->dist_to_origin[i - 1] > this->dist_to_origin[i])
//            {
//                cur_start = i;
//                dist = this->dist_to_origin[i - 1];
//            }
//            else if (cur_start != -1 && this->dist_to_origin[i] > dist)
//            {
//                for (int j = st + cur_start; j <= st + i; j++)
//                    this->is_obstacle[j] = true;
//                cur_start = -1;
//            }
//            if (i == n - 1 && cur_start != -1)
//            {
//                for (int j = st + cur_start; j <= st + i; j++)
//                    this->is_obstacle[j] = true;
//            }
//            i++;
//        }
//    }
//    return is_obstacle;
//}

float Boundary_detection::distance_to_line(cv::Point2f p1, cv::Point2f p2)
{
    float a = p1.y - p2.y;
    float b = p1.x - p2.x;
    float c = p1.x * p2.y - p2.x * p1.y;
    if (a == 0 && b == 0)
        cout << "Error\n";
    return std::abs(c) / std::sqrt(a * a + b * b);
}

int Boundary_detection::get_index_horizontal(int idx)
{
     
}

#if USE_MULTIPLE_LIDAR
void Boundary_detection::find_boundary_from_half_scan(int scan_id, int k, bool masking)
{
    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
    int n = ed - st;
    if (n == 0) return;
    if (n - (2 * k) < 0) return;
    
    // elevation filter
    auto is_elevating = elevation_filter_abs(scan_id);
    
    std::vector<bool> edge_start_CW(is_elevating.size(), false); 
    std::vector<bool> edge_end_CW(is_elevating.size(), false);
    std::vector<bool> edge_start_CCW(is_elevating.size(), false); 
    std::vector<bool> edge_end_CCW(is_elevating.size(), false);

    // test is_start_point & is_end_point
    for (int i = 0; i < is_elevating.size(); i++) 
    {
        if (is_end_point(scan_id, i, is_elevating, Boundary_detection::ScanDirection::CLOCKWISE)) 
        {
            edge_end_CW[i] = true;
            this->is_edge_end[st+i] = true;
        }
        if (is_start_point(scan_id, i, is_elevating, Boundary_detection::ScanDirection::CLOCKWISE)) 
        {
            edge_start_CW[i] = true;
            this->is_edge_start[st+i] = true;
        }
        if (is_end_point(scan_id, i, is_elevating, Boundary_detection::ScanDirection::COUNTER_CLOCKWISE)) 
        {
            edge_end_CCW[i] = true;
            this->is_edge_end[st+i] = true;
        }
        if (is_start_point(scan_id, i, is_elevating, Boundary_detection::ScanDirection::COUNTER_CLOCKWISE)) 
        {
            edge_start_CCW[i] = true;
            this->is_edge_start[st+i] = true;
        }
    }
        
    // local min of direction change
    auto is_direction_change = local_min_of_direction_change(scan_id);
    
    // continuous filter
    auto is_continuous = continuous_filter(scan_id);

    // start from one side of the scan and search
    int i = 0;
    int cur_start = 0, cur_end = 0;
    float cur_height = 0;
    float missed_rate, missed = 0.0f;
    while (i < n)
    {
        if (is_changing_angle[st+i] && edge_start_CW[i] && !is_obstacle[st+i])
        {
            this->is_boundary[st+i] = true;
            // cur_start = i;
            // missed = 0.0f;
            // while (i + 1 < n)
            // {
            //     cur_end = i;
            //     if (is_elevating[i] < 1.0f)
            //         missed += 1.0f;
            //     missed_rate = missed / (cur_end - cur_start + 1);
            //     cur_height = this->pointcloud[st + cur_end][2] - this->pointcloud[st + cur_start][2];
            //     if (missed > 10 && missed_rate > 0.5) break;
            //     if (is_obstacle[st+i]) break;
            //     if (cur_height < 0.0f) break;
            //     if (cur_height > 0.05 && edge_end_CW[i])
            //     {
            //         for (int j = cur_start; j <= cur_end; j++)
            //         {
            //             this->is_boundary[st + j] = true;
            //             if (j - cur_start > 2) break;
            //         }
            //         break;
            //     }
            //     if (cur_height > 0.1)
            //     {
            //         for (int j = cur_start; j <= cur_end; j++)
            //         {
            //             this->is_boundary[st + j] = true;
            //             if (j - cur_start > 2) break;
            //         }
            //         break;
            //     }
            //     i++;
            // }
        }
        i++;
    }
    i = n - 1;
    cur_start = 0, cur_end = 0;
    cur_height = 0;
    missed_rate, missed = 0.0f;
    while (i >= 0)
    {
        if (is_changing_angle[st+i] && edge_start_CCW[i] && !is_obstacle[st+i])
        {
            this->is_boundary[st+i] = true;
            // cur_start = i;
            // missed = 0.0f;
            // while (i - 1 >= 0)
            // {
            //     cur_end = i;
            //     if (is_elevating[i] > -1.0f)
            //         missed += 1.0f;
            //     missed_rate = missed / (cur_start - cur_end + 1);
            //     cur_height = this->pointcloud[st + cur_end][2] - this->pointcloud[st + cur_start][2];
            //     if (missed > 10 && missed_rate > 0.5) break;
            //     if (cur_height < 0.0f) break;
            //     if (is_obstacle[st+i]) break;
            //     if (cur_height > 0.05 && edge_end_CCW[i])
            //     {
            //         for (int j = cur_start; j >= cur_end; j--)
            //         {
            //             this->is_boundary[st + j] = true;
            //             if (j - cur_start < -2) break;
            //         }
            //         break;
            //     }
            //     if (cur_height > 0.1)
            //     {
            //         for (int j = cur_start; j >= cur_end; j--)
            //         {
            //             this->is_boundary[st + j] = true;
            //             if (j - cur_start < -2) break;
            //         }
            //         break;
            //     }
            //     i--;
            // }
        }
        i--;
    }
}
#else
void Boundary_detection::find_boundary_from_half_scan(int scan_id, int k, bool masking)
{
    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
    int n = ed - st;
    if (n == 0)
        return;
    if (n - (2 * k) < 0)
        return;

    // elevation filter
    auto is_elevating = elevation_filter(scan_id);
    std::vector<bool> edge_start(is_elevating.size(), false); 
    std::vector<bool> edge_end(is_elevating.size(), false);
    edge_filter_from_elevation(scan_id, is_elevating, edge_start, edge_end);

    // local min of direction change
    auto is_direction_change = local_min_of_direction_change(scan_id);

    // continuous filter
    auto is_continuous = continuous_filter(scan_id);

    // start from one side of the scan and search
    bool found = false;
    if (scan_id % 2 == 0)
    { // left scan
        int i = n - 1;
        int cur_start = 0, cur_end = 0;
        float cur_height = 0;
        float missed_rate, missed = 0.0f;
        while (i >= 0)
        {
            if (is_direction_change[i] && edge_start[i])
            {
                cur_start = i;
                missed = 0.0f;
                while (i - 1 >= 0)
                {
                    if (is_direction_change[i] && edge_start[i] && cur_height < MIN_CURB_HEIGHT)
                    {
                        cur_start = i;
                    }
                    cur_end = i;
                    if (!is_elevating[i])
                        missed += 1.0f;
                    missed_rate = missed / (cur_start - cur_end + 1);
                    cur_height = this->pointcloud[st + cur_end][2] - this->pointcloud[st + cur_start][2];
                    if (!is_continuous[i])
                        break;
                    if (missed > 10 && missed_rate > 0.3)
                        break;
                    if (cur_height > 0.05 && edge_end[i])
                    {
                        for (int j = cur_end; j <= cur_start; j++)
                        {
                            this->is_boundary[st + j] = true;
                        }
                        found = true;
                        break;
                    }
                    if (cur_height > 0.1)
                    {
                        for (int j = cur_end; j <= cur_start; j++)
                        {
                            this->is_boundary[st + j] = true;
                        }
                        found = true;
                        break;
                    }
                    i--;
                }
            }
            i--;
            if (found)
                break;
        }
    }
    else
    {
        int i = 0;
        int cur_start = 0, cur_end = 0;
        float cur_height = 0;
        float missed_rate, missed = 0.0f;
        while (i < n)
        {
            if (is_direction_change[i] && edge_start[i])
            {
                cur_start = i;
                missed = 0.0f;
                while (i + 1 < n)
                {
                    if (is_direction_change[i] && edge_start[i] && cur_height < MIN_CURB_HEIGHT)
                    {
                        cur_start = i;
                    }
                    cur_end = i;
                    if (!is_elevating[i])
                        missed += 1.0f;
                    missed_rate = missed / (cur_end - cur_start + 1);
                    cur_height = this->pointcloud[st + cur_end][2] - this->pointcloud[st + cur_start][2];
                    if (!is_continuous[i])
                        break;
                    if (missed > 10 && missed_rate > 0.3)
                        break;
                    if (cur_height > 0.05 && edge_end[i])
                    {
                        for (int j = cur_start; j <= cur_end; j++)
                        {
                            this->is_boundary[st + j] = true;
                        }
                        found = true;
                        break;
                    }
                    if (cur_height > 0.1)
                    {
                        for (int j = cur_start; j <= cur_end; j++)
                        {
                            this->is_boundary[st + j] = true;
                        }
                        found = true;
                        break;
                    }
                    i++;
                }
            }
            i++;
            if (found)
                break;
        }
    }
}
#endif

void Boundary_detection::detect(const cv::Mat &rot, const cv::Mat &trans, bool vis) {
    pointcloud_preprocessing(rot);
    for (int i = 0; i < 16; i++)
    {
        // auto is_elevating = elevation_filter_abs(i);
        find_boundary_from_half_scan(i, 8, false);
    }
}

std::vector<bool> Boundary_detection::run_detection(bool vis) {
    // Create Viewer
    cv::viz::Viz3d viewer("Velodyne");
    // Register Keyboard Callback
    viewer.registerKeyboardCallback([](const cv::viz::KeyboardEvent &event, void *cookie) {
        // Close Viewer
        if (event.code == 'q' && event.action == cv::viz::KeyboardEvent::Action::KEY_DOWN)
        {
            static_cast<cv::viz::Viz3d *>(cookie)->close();
        }
    }, &viewer);
    // cv::viz::Viz3d viewer2("Velodyne2");
    // // Register Keyboard Callback
    // viewer.registerKeyboardCallback([](const cv::viz::KeyboardEvent &event, void *cookie) {
    //     // Close Viewer
    //     if (event.code == 'q' && event.action == cv::viz::KeyboardEvent::Action::KEY_DOWN)
    //     {
    //         static_cast<cv::viz::Viz3d *>(cookie)->close();
    //     }
    // }, &viewer2);

    if (this->isPCAP) {
        std::vector<cv::Vec3f> temp = {cv::Vec3f(0, 0, 0)};
        velodyne::VLP16Capture capture(this->directory);
        if (!capture.isOpen())
        {
            std::cerr << "Can't open VelodyneCapture." << std::endl;
            return {};
        }
        while(capture.isRun() && !viewer.wasStopped()){
            high_resolution_clock::time_point start = high_resolution_clock::now();
/*             while (!this->firstRun && true) {
                temp = this->radar_pointcloud;
                if (!temp.empty() && this->secondRun) {
                    if (isnanf(temp.front()[0])) {
                        usleep(10000);
                    } else {
                        this->secondRun = false;
                        break;
                    }
                } else if (temp.empty()) {
                    usleep(10000);     
                } else {
                    break;
                }
            } */
            std::vector<velodyne::Laser> lasers;
            capture >> lasers;
            if (lasers.empty())
            {
                continue;
            }
            high_resolution_clock::time_point t1 = high_resolution_clock::now();
            laser_to_cartesian(lasers);
            pointcloud_preprocessing();
            for (int i = 0; i < 32; i++)
            {
                find_boundary_from_half_scan(i, 8, false);
            }
            // if (vis) update_viewer(this->pointcloud, this->is_boundary, leftLine, rightLine, viewer, this->isPCAP);
            
            auto duration = duration_cast<milliseconds>(high_resolution_clock::now() - t1).count();
            std::cout << "Lidar Time: " << duration << std::endl;
            t1 = high_resolution_clock::now();
            int display_duration = 0;
            /* if (!firstRun) {
                temp = this->radar_pointcloud;
                this->fuser.addRadarData(temp);
            } */
            auto fusionStart = high_resolution_clock::now();
            std::vector<std::vector<cv::Vec3f>> buffers = getLidarBuffers(this->pointcloud, this->is_boundary);
            std::vector<cv::viz::WLine> WLine = this->fuser.displayLidarLine(buffers[1]);
            std::vector<cv::viz::WText3D> confidences = this->fuser.displayConfidence(buffers[1]);
            std::vector<cv::viz::WPolyLine> thirdOrder = this->fuser.displayThirdOrder(buffers[1]);
            std::cout << "Fusion Time: " << duration_cast<milliseconds>(high_resolution_clock::now() - fusionStart).count() << std::endl;
            duration = duration_cast<milliseconds>(high_resolution_clock::now() - t1).count();
            t1 = high_resolution_clock::now();
            int timeRemaining = std::max(30 - int(duration), 10);
            while (true) {
                if (timeRemaining <= display_duration) {
                    break;
                } else {
                    display_duration = int(duration_cast<milliseconds>(high_resolution_clock::now() - t1).count());
                    if (vis) update_viewer(buffers, WLine, confidences, temp, thirdOrder, viewer);
                }    
            }
            std::cout << "Total Time: " << duration_cast<milliseconds>(high_resolution_clock::now() - start).count() << std::endl;
            if (this->firstRun) {
                this->firstRun = false;
                this->secondRun = true;
            }
        }
    }
    viewer.close();
    return {};
}

void Boundary_detection::reset()
{
    this->dist_to_origin = get_dist_to_origin();
    this->theoretical_dist = get_theoretical_dist();
    // this->is_boundary = std::vector<bool>(this->pointcloud.size(), false);
    this->is_boundary_masking = std::vector<bool>(this->pointcloud.size(), false);
    this->is_continuous = std::vector<bool>(this->pointcloud.size(), true);
    this->is_elevating = std::vector<int>(this->pointcloud.size(), false);
    this->is_changing_angle = std::vector<bool>(this->pointcloud.size(), false);
    this->is_local_min = std::vector<bool>(this->pointcloud.size(), false);
    this->is_edge_start = std::vector<bool>(this->pointcloud.size(), false);
    this->is_edge_end = std::vector<bool>(this->pointcloud.size(), false);
    this->is_objects = std::vector<bool>(this->pointcloud.size(), false);
}

string Boundary_detection::get_filename_pointcloud(const string &root_dir, int frame_idx)
{
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(10) << frame_idx;
    string filename = "/home/rtml/LiDAR_camera_calibration_work/data/data_raw/synced/" + root_dir + "velodyne_points/data/" + ss.str() + ".bin";
    return filename;
}

string Boundary_detection::get_filename_image(const string &root_dir, int frame_idx)
{
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(10) << frame_idx;
    string filename = "/home/rtml/LiDAR_camera_calibration_work/data/data_raw/synced/" + root_dir + "image_01/data/" + ss.str() + ".png";
    return filename;
}

void Boundary_detection::print_pointcloud(const vector<vector<float>> &pointcloud)
{
    std::vector<int> count(16, 0);
    int cnt = 0;
    for (auto &p : pointcloud)
    {
        cout << p[0] << " " << p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[6] << endl;
        count[(int)p[4]]++;
    }
    for (auto &c : count)
        cout << c << " ";
    cout << endl;
}

std::vector<std::vector<float>>& Boundary_detection::get_pointcloud() {
    return this->pointcloud; 
}

#if USE_MULTIPLE_LIDAR
std::vector<int>& Boundary_detection::get_result() {
    return this->is_elevating;
}
std::vector<bool>& Boundary_detection::get_result_bool() {
    return this->is_boundary;
}
#else
std::vector<bool>& Boundary_detection::get_result() {
    return this->is_boundary;
}
#endif

void Boundary_detection::timedFunction(std::function<void(void)> func, unsigned int interval) {
    std::thread([func, interval]() {
        while (true) {
            auto x = std::chrono::steady_clock::now() + std::chrono::milliseconds(interval);
            func();
            std::this_thread::sleep_until(x);
        }
    }).detach();
}

void Boundary_detection::expose() {
    this->mem_mutex.lock();
    boost::interprocess::managed_shared_memory segment{boost::interprocess::open_only, "radar_vector"};
    radar_shared *shared = segment.find<radar_shared>("radar_shared").first;
    this->radar_pointcloud.assign(shared->begin(), shared->end());
    this->mem_mutex.unlock();
}

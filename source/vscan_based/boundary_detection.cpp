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

bool Boundary_detection::isRun()
{
    return dataReader.isRun();
}

void Boundary_detection::retrieveData()
{
    dataReader >> pointcloud; 
}

std::vector<cv::viz::WPolyLine> Boundary_detection::getThirdOrderLines(std::vector<cv::Vec3f> &buf)
{
    return fuser.displayThirdOrder(buf);
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

// void Boundary_detection::pointcloud_preprocessing(const cv::Mat &rot)
// {
//     rotate_and_translate_multi_lidar_yaw(rot);
//     reset();
//     // auto t_start = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
//     // auto t_end = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - t_start;
// }

void Boundary_detection::pointcloud_preprocessing(const cv::Mat &rot)
{
    rotate_and_translate_multi_lidar_yaw(rot);
    max_height_filter(.45);
    rearrange_pointcloud();
    reset();
}

std::vector<float> Boundary_detection::get_dist_to_origin() {
    std::vector<float> dist(pointcloud.size());
    for (int i = 0; i < dist.size(); i++) 
        dist[i] = std::sqrt(pointcloud[i][0]*pointcloud[i][0] + pointcloud[i][1]*pointcloud[i][1] + pointcloud[i][2]*pointcloud[i][2]); 
    return dist;
}

float Boundary_detection::dist_between(const std::vector<float> &p1, const std::vector<float> &p2) {
    return std::sqrt((p2[0]-p1[0])*(p2[0]-p1[0]) + (p2[1]-p1[1])*(p2[1]-p1[1]) + (p2[2]-p1[2])*(p2[2]-p1[2]));
}

std::vector<bool> Boundary_detection::continuous_filter(int scan_id) {
    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
    int n = ed - st;
    std::vector<bool> is_continuous(n, true);
    std::vector<float> thres(this->dist_to_origin.begin()+st, this->dist_to_origin.begin()+ed);
    for (auto& t : thres) t *= THETA_R * 7;
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

float Boundary_detection::get_angle(const std::vector<float> &v1, const std::vector<float> &v2)
{
    float angle;
    float dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    float mag_1 = std::sqrt(v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2]);
    float mag_2 = std::sqrt(v2[0] * v2[0] + v2[1] * v2[1] + v2[2] * v2[2]);
    return std::acos(dot_product / (mag_1 * mag_2)) * 180.0 / PI;
}

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

std::vector<bool> get_local_min(const std::vector<float> &vec) {
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
            thres_z = 0.002;
        if (z_diff[i] > thres_z)
            is_elevate[i] = 1;
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

void Boundary_detection::edge_filter_from_elevation(int scan_id, const std::vector<int> &elevation, std::vector<bool> &edge_start, std::vector<bool>& edge_end) {
    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
    int n = ed - st;
    int k = 7;
    std::vector<int> f_start, f_end;
    if (n <= (2 * k)) return; 
    if (scan_id % 2 == 0) { // left scan
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
                        if (masking) {
                            bool is_masked = false;
                            for (int j = cur_end; j <= cur_start; j++)
                            {
                                if (this->is_objects[st + j]) 
                                {
                                    is_masked = true;
                                    // cout << "Result being masked at " << scan_id / 2 << "\n";
                                    break;
                                }
                            }
                            if (!is_masked) {
                                for (int j = cur_end; j <= cur_start; j++)
                                {
                                    this->is_boundary_masking[st + j] = true;
                                }
                                found = true;
                                break;
                            }
                        }
                        else {
                            for (int j = cur_end; j <= cur_start; j++)
                            {
                                this->is_boundary[st + j] = true;
                            }
                            found = true;
                            break;
                        }
                    }
                    if (cur_height > 0.1)
                    {
                        if (masking) {
                            bool is_masked = false;
                            for (int j = cur_end; j <= cur_start; j++)
                            {
                                if (this->is_objects[st + j]) 
                                {
                                    is_masked = true;
                                    // cout << "Result being masked at " << scan_id / 2 << "\n";
                                    break;
                                }
                            }
                            if (!is_masked) {
                                for (int j = cur_end; j <= cur_start; j++)
                                {
                                    this->is_boundary_masking[st + j] = true;
                                }
                                found = true;
                                break;
                            }
                        }
                        else {
                            for (int j = cur_end; j <= cur_start; j++)
                            {
                                this->is_boundary[st + j] = true;
                            }
                            found = true;
                            break;
                        }
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
                        if (masking) {
                            bool is_masked = false;
                            for (int j = cur_start; j <= cur_end; j++)
                            {
                                if (this->is_objects[st + j]) 
                                {
                                    is_masked = true;
                                    // cout << "Result being masked at " << scan_id / 2 << "\n";
                                    break;
                                }
                            }
                            if (!is_masked) {
                                for (int j = cur_start; j <= cur_end; j++)
                                {
                                    this->is_boundary_masking[st + j] = true;
                                }
                                found = true;
                                break;
                            }
                        }
                        else {
                            for (int j = cur_start; j <= cur_end; j++)
                            {
                                this->is_boundary[st + j] = true;
                            }
                            found = true;
                            break;
                        }
                    }
                    if (cur_height > 0.1)
                    {
                        if (masking) {
                            bool is_masked = false;
                            for (int j = cur_start; j <= cur_end; j++)
                            {
                                if (this->is_objects[st + j]) 
                                {
                                    is_masked = true;
                                    // cout << "Result being masked at " << scan_id / 2 << "\n";
                                    break;
                                }
                            }
                            if (!is_masked) {
                                for (int j = cur_start; j <= cur_end; j++)
                                {
                                    this->is_boundary_masking[st + j] = true;
                                }
                                found = true;
                                break;
                            }
                        }
                        else {
                            for (int j = cur_start; j <= cur_end; j++)
                            {
                                this->is_boundary[st + j] = true;
                            }
                            found = true;
                            break;
                        }
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

void Boundary_detection::detect(const cv::Mat &rot, const cv::Mat &trans) 
{
    for (int i = 0; i < 32; i++)
    {
        find_boundary_from_half_scan(i, 8, false);
    }
    // If radar data available, read the data from shared memory

}

void Boundary_detection::reset()
{
    this->dist_to_origin = get_dist_to_origin();
    this->is_boundary = std::vector<bool>(this->pointcloud.size(), false);
    this->is_boundary_masking = std::vector<bool>(this->pointcloud.size(), false);
    this->is_continuous = std::vector<bool>(this->pointcloud.size(), true);
    this->is_elevating = std::vector<bool>(this->pointcloud.size(), false);
    this->is_changing_angle = std::vector<bool>(this->pointcloud.size(), false);
    this->is_local_min = std::vector<bool>(this->pointcloud.size(), false);
    this->is_edge_start = std::vector<bool>(this->pointcloud.size(), false);
    this->is_edge_end = std::vector<bool>(this->pointcloud.size(), false);
    this->is_objects = std::vector<bool>(this->pointcloud.size(), false);
}

std::vector<std::vector<float>>& Boundary_detection::get_pointcloud() 
{
    return pointcloud; 
}

std::vector<int> Boundary_detection::get_result() 
{
    return is_boundary_int;
}

std::vector<bool> Boundary_detection::get_result_bool() 
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

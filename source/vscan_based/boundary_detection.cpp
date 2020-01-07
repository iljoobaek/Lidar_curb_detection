#include "boundary_detection.h"
using namespace std::chrono;

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
    std::reverse(pointcloud.begin(), pointcloud.end());
    pointcloud_raw = pointcloud;
}

std::vector<cv::viz::WPolyLine> Boundary_detection::getThirdOrderLines(std::vector<cv::Vec3f> &buf)
{
    return fuser.displayThirdOrder(buf);
}

std::vector<float> Boundary_detection::getLeftBoundaryCoeffs()
{
    return fuser.getLeftCoeffs();
}

std::vector<float> Boundary_detection::getRightBoundaryCoeffs()
{
    return fuser.getRightCoeffs();
}

void Boundary_detection::writeResultTotxt(const std::vector<float> &boundaryCoeffs, int leftRight)
{
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(10) << currentFrameIdx;
    std::string fn = "detection_result/" + ss.str();
    fn += (leftRight == 0) ? "_l.txt" : "_r.txt";

    std::ofstream file_out;
    file_out.open(fn);   
    if (boundaryCoeffs.empty()) {
        file_out << "null\n";
    }
    else {
        std::stringstream ss;
        ss << boundaryCoeffs[0] << " " << boundaryCoeffs[1] << " " << boundaryCoeffs[2] << " "  << boundaryCoeffs[3] << "\n";
        file_out << ss.str();
        for (int i = 0; i < 32; i++)
        {
            if (i % 2 == leftRight) // left or right scan
            {
                for (int j = this->ranges[i][0]; j < this->ranges[i][1]; j++)
                {
                    if (is_boundary[j])
                    {
                        ss.str("");
                        ss.clear();
                        ss << this->pointcloud[j][0] << " " << this->pointcloud[j][1] << " " << this->pointcloud[j][2] << "\n";
                        file_out << ss.str();
                    }
                }
            }
        }
    }
    file_out.close();
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
    index_mapping.clear();
    index_mapping.reserve(pointcloud.size());
    int cur = 0;
    for (int i = 0; i < this->pointcloud.size(); i++)
    { 
        if (this->pointcloud[i][2] < max_height)
        {
            index_mapping.push_back(i);
            this->pointcloud[cur] = this->pointcloud[i];
            cur++;
        }
    }
    this->pointcloud.erase(this->pointcloud.begin() + cur, this->pointcloud.end());
}

void Boundary_detection::rearrange_pointcloud() 
{
    std::vector<std::vector<float>> pointcloud_copy(this->pointcloud.begin(), this->pointcloud.end());
    std::vector<int> index_mapping_copy(index_mapping.begin(), index_mapping.end());
    int cur_idx = 0;
    for (int i = 0; i < num_of_scan; i++)
    {
        this->ranges[i * 2][0] = cur_idx;
        auto iter = pointcloud_copy.begin();
        while ((iter = std::find_if(iter, pointcloud_copy.end(), [&](const std::vector<float> &point) { return point[4] == static_cast<float>(i) && point[6] > 0; })) != pointcloud_copy.end())
        {
            index_mapping[cur_idx] = index_mapping_copy[(iter-pointcloud_copy.begin())];
            this->pointcloud[cur_idx++] = (*iter);
            iter++;
        }
        this->ranges[i * 2][1] = cur_idx;
        this->ranges[i * 2 + 1][0] = cur_idx;
        iter = pointcloud_copy.begin();
        while ((iter = std::find_if(iter, pointcloud_copy.end(), [&](const std::vector<float> &point) { return point[4] == static_cast<float>(i) && point[6] <= 0; })) != pointcloud_copy.end())
        {
            index_mapping[cur_idx] = index_mapping_copy[(iter-pointcloud_copy.begin())];
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
        if (z_diff[i] > 0.0)
            is_elevate[i] = 1;
    }
    // std::vector<int> filter({1,1,1,1,1,1,1,1,1});
    // auto res = conv(is_elevate, filter);
    // for (int i = 0; i < is_elevate.size(); i++)
    // {
    //     if (res[i + 4] >= 4)
    //         is_elevate[i] = 1;
    //     else
    //         is_elevate[i] = 0;
    // }
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

std::vector<std::vector<cv::Vec3f>> Boundary_detection::runDetection(const cv::Mat &rot, const cv::Mat &trans) 
{
    std::string fn_image = get_filename_image(root_path, data_folder, currentFrameIdx);
    std::cout << fn_image << std::endl;
    cv::Mat img = cv::imread(fn_image);
    // if (find_objects_from_image(fn_image, img))
    // {   
    //     std::cout << "--- moving objects detected---\n";
    // }
    // else
    // {
    //     std::cout << "--- no objects detected---\n";
    // }
    // cv::resize(img, img, cv::Size(img.cols/2, img.rows/2));
    cv::imshow("image", img);
    cv::waitKey(1);
    
    for (int i = 0; i < num_of_scan*2; i++)
    {
        find_boundary_from_half_scan(i, 8, false);
    }
    // If radar data available, read the data from shared memory

    // is_boundary = std::vector<bool>(pointcloud.size(), false);
    auto buf = getLidarBuffers(pointcloud, is_boundary);

    // std::vector<float> leftBoundaryCoeffs = getLeftBoundaryCoeffs();;
    // std::vector<float> rightBoundaryCoeffs = getRightBoundaryCoeffs();;
    // writeResultTotxt(leftBoundaryCoeffs, 0);
    // writeResultTotxt(rightBoundaryCoeffs, 1);
    currentFrameIdx++;
    return buf;
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

std::vector<std::vector<float>> Boundary_detection::get_image_points()
{
    std::vector<std::vector<float>> image_points(pointcloud.size(), std::vector<float>(2));
    for (int i = 0; i < image_points.size(); i++)
    {
        float scale = lidar_to_image[2][0] * pointcloud_raw[index_mapping[i]][0] + lidar_to_image[2][1] * pointcloud_raw[index_mapping[i]][1] + lidar_to_image[2][2] * pointcloud_raw[index_mapping[i]][2] + lidar_to_image[2][3];
        image_points[i][0] = (lidar_to_image[0][0] * pointcloud_raw[index_mapping[i]][0] + lidar_to_image[0][1] * pointcloud_raw[index_mapping[i]][1] + lidar_to_image[0][2] * pointcloud_raw[index_mapping[i]][2] + lidar_to_image[0][3]) / scale;
        image_points[i][1] = (lidar_to_image[1][0] * pointcloud_raw[index_mapping[i]][0] + lidar_to_image[1][1] * pointcloud_raw[index_mapping[i]][1] + lidar_to_image[1][2] * pointcloud_raw[index_mapping[i]][2] + lidar_to_image[1][3]) / scale;
    }
    return image_points;
}

bool Boundary_detection::is_in_bounding_box(const std::vector<float> &img_point, const std::vector<std::vector<int>> &bounding_boxes)
{
    for (auto &box : bounding_boxes)
    {
        if (box[1] <= img_point[0] && img_point[0] <= box[3] && box[0] <= img_point[1] && img_point[1] <= box[2])
        {
            return true;
        }
    }
    return false;
}

bool Boundary_detection::find_objects_from_image(std::string filename, cv::Mat &img)
{
    PyObject *data = object_detector->call_method("run", filename);
    std::vector<std::vector<int>> bounding_boxes;
    if (data)
    {
        auto box = object_detector->listTupleToVector(data);
        for (int i = 0; i < box.size() / 4; i++)
        {
            bounding_boxes.push_back({box[i * 4] * object_detector->ROI_height + object_detector->ROI_offset_y, box[i * 4 + 1] * object_detector->img_width, box[i * 4 + 2] * object_detector->ROI_height + object_detector->ROI_offset_y, box[i * 4 + 3] * object_detector->img_width});
        }
    }
    else
    {
        return false;
    }
    auto image_points = get_image_points();
    for (auto &box : bounding_boxes)
        cv::rectangle(img, cv::Point(box[1], box[0]), cv::Point(box[3], box[2]), cv::Scalar(255, 255, 255));
    int cnt = 0;
    for (int i = 0; i < image_points.size(); i++)
    {
        if (is_in_bounding_box(image_points[i], bounding_boxes))
        {
            cv::circle(img, cv::Point((int)image_points[i][0], (int)image_points[i][1]), 1, cv::Scalar(0, 255, 0));
            this->is_objects[i] = true;
            cnt++;
        }
        else
        {
            cv::circle(img, cv::Point((int)image_points[i][0], (int)image_points[i][1]), 1, cv::Scalar(255, 0, 0));
            this->is_objects[i] = false;
        }
    }
    std::cout << "Found " << cnt << " points from " << image_points.size() << std::endl;
    return true;
}

std::string Boundary_detection::get_filename_image(std::string root_dir, std::string folder, int frame_idx)
{
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(10) << frame_idx;
    std::string filename = root_dir + folder + "image_01/data/" + ss.str() + ".png";
    return filename;
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
    // std::vector<bool> test;
    // for (auto &point : pointcloud)
    // {
    //     if (point[4] == num_of_scan / 2)
    //     {
    //         test.push_back(true);
    //     }
    //     else
    //     {
    //         test.push_back(false);
    //     }
    // }
    
    // std::vector<bool> test(pointcloud.size(), false);
    // for (int i = 0; i < ranges.size(); i++)
    // {
    //     if (i % 2 == 0 && ranges[i][0] != ranges[i][1])
    //     {
    //         // test[ranges[i][0]] = true;
    //         int idx = ranges[i][0];
    //         for (int j = 0; j < 5; j++)
    //         {
    //             if (idx + j >= 0 && idx + j < pointcloud.size())
    //             {
    //                 test[idx+j] = true;
    //             }
    //         }
    //     }
    // }
    // return test;
    
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

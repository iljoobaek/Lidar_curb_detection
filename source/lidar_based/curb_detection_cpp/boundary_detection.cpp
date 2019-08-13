#include "boundary_detection.h"

template<typename T>
std::vector<T> conv(std::vector<T> const &f, std::vector<T> const &g) {
    int const nf = f.size();
    int const ng = g.size();
    int const n  = nf + ng - 1;
    std::vector<T> out(n, T());
    for(auto i(0); i < n; ++i) {
        int const jmn = (i >= ng - 1)? i - (ng - 1) : 0;
        int const jmx = (i <  nf - 1)? i            : nf - 1;
        for(auto j(jmn); j <= jmx; ++j) {
            out[i] += (f[j] * g[i - j]);
        }
    }
    return out; 
}

void update_viewer(const std::vector<std::vector<float>>& pointcloud, const std::vector<bool>& result, std::vector<cv::Vec3f> radar_pointcloud, cv::viz::Viz3d viewer) {
    std::vector<cv::Vec3f> buffer(pointcloud.size());
    std::vector<cv::Vec3b> colors(pointcloud.size());
    for (int i = 0; i < pointcloud.size(); i++) {
        buffer[i] = cv::Vec3f(pointcloud[i][0], pointcloud[i][1], pointcloud[i][2]);
        if (result[i]) colors[i] = {0,0,255};
        else colors[i] = {255,255,255};
    } 
    if (buffer.empty()) {return;}
    // Create Widget
    cv::Mat cloudMat = cv::Mat(static_cast<int>(buffer.size()), 1, CV_32FC3, &buffer[0]);
    cv::Mat colorMat = cv::Mat(static_cast<int>(colors.size()), 1, CV_8UC3, &colors[0]);

    cv::Mat radarMat = cv::Mat(static_cast<int>(radar_pointcloud.size()),
        1, CV_32FC3, &radar_pointcloud[0]);

    cv::viz::WCloudCollection collection;
    collection.addCloud(radarMat, cv::viz::Color::red());
    collection.addCloud(cloudMat, colorMat);
    // Show Point Cloudcloud

    viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(2));
    viewer.showWidget("Cloud", collection);
    viewer.spinOnce();
}

std::vector<std::vector<float>> Boundary_detection::read_bin(string filename) {
    std::vector<std::vector<float>> pointcloud;
    int32_t num = 1000000;
    float* data = (float*)malloc(num*sizeof(float));
    float *px = data, *py = data+1, *pz = data+2, *pi = data+3, *pr = data+4;

    FILE *stream = fopen(filename.c_str(), "rb");
    num = fread(data, sizeof(float), num, stream) / 5;
    for (int32_t i = 0; i < num; i++) {
        float dist = std::sqrt((*px)*(*px) + (*py)*(*py) + (*pz)*(*pz));
        float theta = std::atan2(*py, *px) * 180.0f / PI;
        if (dist > 0.9f && *px >= 0.0f) pointcloud.push_back({*px, *py, *pz, *pi, *pr, dist, theta});
        px += 5, py += 5, pz += 5, pi += 5, pr += 5;
    }
    fclose(stream);
    cout << "Read in " << pointcloud.size() << " points\n"; 
    return pointcloud;
}

void Boundary_detection::rotate_and_translate() {
    if (this->pointcloud.empty()) return;
    // rotation matrix
    // [cos(theta),   0, sin(theta)]
    // [0,            1,          0]
    // [-sin(theta),  0, cos(theta)]
    float theta = this->tilted_angle * PI / 180.0f;
    // cout << "[ "<< std::cos(theta) << " " << 0.0f << " " << std::sin(theta) << "\n";
    // cout << 0.0f << " " << 1.0f << " " << 0.0f << "\n";
    // cout << -std::sin(theta) << " " << 0.0f << " " << std::cos(theta) << " ]"<< "\n";
    for (auto& point : this->pointcloud) {
        float x = point[0] * std::cos(theta) + point[2] * std::sin(theta);
        float z = point[0] * (-std::sin(theta)) + point[2] * std::cos(theta) + this->sensor_height;
        point[0] = x;
        point[2] = z;
    }
}

void Boundary_detection::max_height_filter(float max_height) {
    int cur = 0;
    for (int i = 0; i < this->pointcloud.size(); i++) {
        if (this->pointcloud[i][2] < max_height) this->pointcloud[cur++] = this->pointcloud[i];
    }
    this->pointcloud.erase(this->pointcloud.begin()+cur, this->pointcloud.end());
}

void Boundary_detection::reorder_pointcloud() {
    int idx = 0;
    for (auto& r : ranges) {
        bool flag = false;
        float theta_prev = 100.0f;
        // cout << "--------- scan " << idx++ << " -----------" << endl;
        cout << r[0] << " " << r[1] << endl;
        for (int i = r[0]; i < r[1]; i++) {
            // cout << this->pointcloud[i][6] << endl;
            // if (this->pointcloud[i][6] > theta_prev) {
            //     flag = true;
            //     cout << "theta_prev: " << theta_prev << endl;
            // }
            // if (flag) cout << this->pointcloud[i][6] << endl;
            theta_prev = this->pointcloud[i][6]; 
        }
        break; 
        // cout << theta[i] << endl;
    }
    // std::sort(this->pointcloud.begin(), this->pointcloud.end(), );
}

void Boundary_detection::rearrange_pointcloud() {
    std::vector<std::vector<float>> pointcloud_copy(this->pointcloud.begin(), this->pointcloud.end());
    int cur_idx = 0;
    for (int i = 0; i < num_of_scan; i++) {
        this->ranges[i*2][0] = cur_idx;
        auto iter = pointcloud_copy.begin();
        while ((iter = std::find_if(iter, pointcloud_copy.end(), [&](const vector<float>& point){return point[4] == static_cast<float>(i) && point[6] > 0;})) != pointcloud_copy.end()) {
            this->pointcloud[cur_idx++] = (*iter);
            iter++;
        }
        this->ranges[i*2][1] = cur_idx;
        this->ranges[i*2+1][0] = cur_idx;
        iter = pointcloud_copy.begin();
        while ((iter = std::find_if(iter, pointcloud_copy.end(), [&](const vector<float>& point){return point[4] == static_cast<float>(i) && point[6] <= 0;})) != pointcloud_copy.end()) {
            this->pointcloud[cur_idx++] = (*iter);
            iter++;
        }
        this->ranges[i*2+1][1] = cur_idx;
    }
    assert(cur_idx == this->pointcloud.size());
}

void Boundary_detection::rearrange_pointcloud_sort() {
    std::sort(this->pointcloud.begin(), this->pointcloud.end(), 
        [](const std::vector<float>& p1, const std::vector<float>& p2){
            if (p1[4] == p2[4]) return p1[6] > p2[6];
            else return p1[4] < p2[4];});
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

void Boundary_detection::pointcloud_preprocessing() {
    rotate_and_translate();
    max_height_filter(.45);
    rearrange_pointcloud();
    reset();
}

float Boundary_detection::dist_between(const std::vector<float>& p1, const std::vector<float>& p2) {
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
            is_continuous[i+1] = false;
            this->is_continuous[st+i] = false;
            this->is_continuous[st+i+1] = false;
        }
    }
    return is_continuous;
}

float Boundary_detection::get_angle(const vector<float>& v1, const vector<float>& v2) {
    float angle;
    float dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    float mag_1 = std::sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]); 
    float mag_2 = std::sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]); 
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
    for (int i = 0; i < n-k; i++) {
        for (int j = i+1; j <= i+k; j++) {
            direction_vecs_right[i][0] = this->pointcloud[st+j][0] - this->pointcloud[st+i][0]; 
            direction_vecs_right[i][1] = this->pointcloud[st+j][1] - this->pointcloud[st+i][1]; 
            direction_vecs_right[i][2] = this->pointcloud[st+j][2] - this->pointcloud[st+i][2]; 
        }
    }
    for (int i = n-1; i >= k; i--) {
        for (int j = i-1; j >= i-k; j--) {
            direction_vecs_left[i][0] = this->pointcloud[st+j][0] - this->pointcloud[st+i][0]; 
            direction_vecs_left[i][1] = this->pointcloud[st+j][1] - this->pointcloud[st+i][1]; 
            direction_vecs_left[i][2] = this->pointcloud[st+j][2] - this->pointcloud[st+i][2]; 
        }
    }
    for (int i = k; i < n-k; i++) {
        angles[i] = get_angle(direction_vecs_left[i], direction_vecs_right[i]);
        if (angles[i] < 150.0) this->is_changing_angle[st+i] = true;
    } 
    return angles;
}

std::vector<bool> get_local_min(const vector<float>& vec) {
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
        this->is_local_min[st+i] = direction_change[i] && local_min[i];
    }
    return direction_change_local_min;
}

std::vector<int> Boundary_detection::elevation_filter(int scan_id) {
    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
    int n = ed - st;
    std::vector<int> is_elevate(n, 0);
    std::vector<float> z_diff(n, 0.0f); 
    float thres_z = 0.005;

    if (scan_id % 2 == 0) { // left scan
        for (int i = n-2; i >= 0; i--) {
            z_diff[i] = this->pointcloud[st+i][2] - this->pointcloud[st+i+1][2];
        }
    }
    else {
        for (int i = 1; i < n; i++) {
            z_diff[i] = this->pointcloud[st+i][2] - this->pointcloud[st+i-1][2];
        }
    }
    for (int i = 0; i < n; i++) {
        if (z_diff[i] > thres_z) is_elevate[i] = 1;
    }
    std::vector<int> filter({1,1,1,1,1,1,1,1,1});
    auto res = conv(is_elevate, filter);
    for (int i = 0; i < is_elevate.size(); i++) {
        if (res[i+4] >= 4) is_elevate[i] = 1;
        else is_elevate[i] = 0;
    }
    for (int i = 0; i < is_elevate.size(); i++) {
        if (is_elevate[i] > 0) this->is_elevating[st+i] = true;
    }
    return is_elevate;
}

void Boundary_detection::edge_filter_from_elevation(int scan_id, const std::vector<int>& elevation, std::vector<bool>& edge_start, std::vector<bool>& edge_end) {
    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
    int n = ed - st;
    int k = 7;
    std::vector<int> f_start, f_end;
    if (n <= (2 * k)) return; 
    if (scan_id % 2 == 0) { // left scan
        f_start = {-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1};
        f_end = {1,1,1,1,1,1,1,1,0,0,0,0,0,0,0};
    }
    else {
        f_start = {1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1};
        f_end = {0,0,0,0,0,0,0,1,1,1,1,1,1,1,1};
    }
    auto edge_start_cnt = conv(elevation, f_start);
    auto edge_end_cnt = conv(elevation, f_end);
    for (int i = 0; i < n; i++) {
        if (edge_start_cnt[i+k] >= 2) {
            edge_start[i] = true;
            this->is_edge_start[st+i] = true;
        }
        if (edge_end_cnt[i+k] >= 6) {
            edge_end[i] = true;
            this->is_edge_end[st+i] = true;
        }
    }
}

std::vector<bool> Boundary_detection::obstacle_extraction(int scan_id) {
    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
    int n = ed - st;
    std::vector<bool> is_obstacle(n, false);
    float height_thres = 0.3, dist = 0.0f;
    int ring = scan_id / 2;
    if (scan_id % 2 == 0) { // left scan
        int i = n - 1, cur_start = -1;
        while (i >= 0) {
            if (i == n - 1 && this->pointcloud[st+i][2] > height_thres) {
                cur_start = i;
                dist = this->theoretical_dist[ring];
            }
            else if (cur_start == -1 && i + 1 < n && !this->is_continuous[i+1] && !this->is_continuous[i] && this->dist_to_origin[i+1] > this->dist_to_origin[i]) {
                cur_start = i;
                dist = this->dist_to_origin[i+1];
            }
            else if (cur_start != -1 && this->dist_to_origin[i] > dist) {
                for (int j = st+i+1; j <= st+cur_start; j++) this->is_obstacle[j] = true;
                cur_start = -1;
            }
            if (i == 0 && cur_start != -1) {
                for (int j = st+i+1; j <= st+cur_start; j++) this->is_obstacle[j] = true;
            }
            i--; 
        }
    }
    else {
        int i = 0, cur_start = -1;
        while (i < n) {
            if (i == 0 && this->pointcloud[st+i][2] > height_thres) {
                cur_start = i;
                dist = this->theoretical_dist[ring];
            }
            else if (cur_start == -1 && i - 1 >= 0 && !this->is_continuous[i-1] && !this->is_continuous[i] && this->dist_to_origin[i-1] > this->dist_to_origin[i]) {
                cur_start = i;
                dist = this->dist_to_origin[i-1];
            }
            else if (cur_start != -1 && this->dist_to_origin[i] > dist) {
                for (int j = st+cur_start; j <= st+i; j++) this->is_obstacle[j] = true;
                cur_start = -1;
            }
            if (i == n - 1 && cur_start != -1) {
                for (int j = st+cur_start; j <= st+i; j++) this->is_obstacle[j] = true;
            }
            i++; 
        }
    }
    return is_obstacle;
}

void Boundary_detection::find_boundary_from_half_scan(int scan_id, int k) {
    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
    int n = ed - st;
    if (n == 0) return;
    if (n - (2 * k) < 0) return; 

    // elevation filter
    auto is_elevating = elevation_filter(scan_id);
    std::vector<bool> edge_start(is_elevating.size(), false); 
    std::vector<bool> edge_end(is_elevating.size(), false);
    edge_filter_from_elevation(scan_id, is_elevating, edge_start, edge_end);

    // local min of direction change
    auto is_direction_change = local_min_of_direction_change(scan_id);

    // continuous filter
    auto is_continuous = continuous_filter(scan_id);

    auto is_obstacle = obstacle_extraction(scan_id);

    bool found = false;
    if (scan_id % 2 == 0) { // left scan
        int i = n - 1;
        int cur_start = 0, cur_end = 0;
        float cur_height = 0;
        float missed_rate, missed = 0.0f;
        while (i >= 0) {
            if (is_direction_change[i] && edge_start[i]) {
                cur_start = i;
                missed = 0.0f;
                while (i - 1 >= 0) {
                    if (is_direction_change[i] && edge_start[i] && cur_height < MIN_CURB_HEIGHT) {
                        cur_start = i;
                    }
                    cur_end = i;
                    if (!is_elevating[i]) missed += 1.0f;
                    missed_rate = missed / (cur_start - cur_end + 1);
                    cur_height = this->pointcloud[st+cur_end][2] - this->pointcloud[st+cur_start][2];
                    if (missed > 10 && missed_rate > 0.3) break;
                    if (cur_height > 0.05 && edge_end[i]) {
                        for (int j = cur_end; j <= cur_start; j++) {
                            this->is_boundary[st+j] = true;
                        }
                        found = true;
                        break;
                    }
                    if (cur_height > 0.1) {
                        for (int j = cur_end; j <= cur_start; j++) {
                            this->is_boundary[st+j] = true;
                        }
                        found = true;
                        break;
                    }
                    i--;
                }
            }
            i--;
            if (found) break;
        }    
    }
    else {
        int i = 0;
        int cur_start = 0, cur_end = 0;
        float cur_height = 0;
        float missed_rate, missed = 0.0f;
        while (i < n) {
            if (is_direction_change[i] && edge_start[i]) {
                cur_start = i;
                missed = 0.0f;
                while (i + 1 < n) {
                    if (is_direction_change[i] && edge_start[i] && cur_height < MIN_CURB_HEIGHT) {
                        cur_start = i;
                    }
                    cur_end = i;
                    if (!is_elevating[i]) missed += 1.0f;
                    missed_rate = missed / (cur_end - cur_start + 1);
                    cur_height = this->pointcloud[st+cur_end][2] - this->pointcloud[st+cur_start][2];
                    if (missed > 10 && missed_rate > 0.3) break;
                    if (cur_height > 0.05 && edge_end[i]) {
                        for (int j = cur_start; j <= cur_end; j++) {
                            this->is_boundary[st+j] = true;
                        }
                        found = true;
                        break;
                    }
                    if (cur_height > 0.1) {
                        for (int j = cur_start; j <= cur_end; j++) {
                            this->is_boundary[st+j] = true;
                        }
                        found = true;
                        break;
                    }
                    i++;
                }
            }
            i++;
            if (found) break;
        }    
    }
}

std::vector<bool> Boundary_detection::run_detection(bool vis) {
    // Create Viewer
    cv::viz::Viz3d viewer( "Velodyne" );
    // Register Keyboard Callback
    viewer.registerKeyboardCallback([]( const cv::viz::KeyboardEvent& event, void* cookie ){
        // Close Viewer
        if( event.code == 'q' && event.action == cv::viz::KeyboardEvent::Action::KEY_DOWN ){
            static_cast<cv::viz::Viz3d*>( cookie )->close();
        }
        }
        , &viewer
    );

    if (this->directory == "test1/") {
        std::vector<cv::Vec3f> test;
        for (int i = 25; i < 911; i++) {
            high_resolution_clock::time_point start = high_resolution_clock::now();
            while (true) {
                test = this->radar_pointcloud;
                if (!test.empty() && i == 25) {
                    if (isnanf(test.front()[0])) {
                        usleep(10000);
                    } else {
                        break;
                    }
                } else {
                    break;                    
                }
            }
            high_resolution_clock::time_point t1 = high_resolution_clock::now();
            
            string filename = this->directory + std::to_string(i) + ".bin";
            this->pointcloud = read_bin(filename);
            pointcloud_preprocessing();
            reset();
            for (int i = 0; i < 32; i++) {
                find_boundary_from_half_scan(i, 8);
            }

            high_resolution_clock::time_point t2 = high_resolution_clock::now();
            auto duration = duration_cast<milliseconds>(t2 - t1).count();
            cout << duration << endl;
            int timeRemaining = 90 - int(duration);
            std::cout << timeRemaining << std::endl;

            t1 = high_resolution_clock::now();
            int display_duration = 0;
            while (true) {
                if (timeRemaining <= display_duration || timeRemaining == 100) {
                    break;
                } else {
                    t2 = high_resolution_clock::now();
                    display_duration = int(duration_cast<milliseconds>(t2 - t1).count());
                    if (vis) update_viewer(this->pointcloud, this->is_boundary, this->radar_pointcloud, viewer);
                }    
            }
            high_resolution_clock::time_point end = high_resolution_clock::now();
            auto duration2 = duration_cast<milliseconds>(end - start).count();
            cout << duration2 << endl;
        }
    }
    else if (this->directory == "test2/") {
        for (int i = 0; i < 1; i++) {
            string filename = this->directory + std::to_string(i) + ".bin";
            this->pointcloud = read_bin(filename);
            pointcloud_preprocessing();
            reset();
            for (int i = 0; i < 32; i++) {
                find_boundary_from_half_scan(i, 8);
            }
            if (vis) update_viewer(this->pointcloud, this->is_boundary, this->radar_pointcloud, viewer);
        }
    }
    return {};
}

void Boundary_detection::reset() {
        this->dist_to_origin = get_dist_to_origin();
        this->theoretical_dist = get_theoretical_dist();
        this->is_boundary = std::vector<bool>(this->pointcloud.size(), false);
        this->is_continuous = std::vector<bool>(this->pointcloud.size(), true);
        this->is_elevating = std::vector<bool>(this->pointcloud.size(), false);
        this->is_changing_angle = std::vector<bool>(this->pointcloud.size(), false);
        this->is_local_min = std::vector<bool>(this->pointcloud.size(), false);
        this->is_edge_start = std::vector<bool>(this->pointcloud.size(), false);
        this->is_edge_end = std::vector<bool>(this->pointcloud.size(), false);
        this->is_obstacle = std::vector<bool>(this->pointcloud.size(), false);
}

void Boundary_detection::print_pointcloud(const std::vector<std::vector<float>>& pointcloud) {
    std::vector<int> count(16, 0);
    int cnt = 0;
    for (auto& p : pointcloud) {
        cout << p[0] << " " <<  p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[6] << endl; 
        count[(int)p[4]]++; 
    }
    for (auto& c : count) cout << c << " ";
    cout << endl;
}

std::vector<std::vector<float>>& Boundary_detection::get_pointcloud() {
    return this->pointcloud; 
}

std::vector<bool>& Boundary_detection::get_result() {
    return this->is_boundary;
}


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
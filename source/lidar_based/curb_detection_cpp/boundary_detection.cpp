#include "boundary_detection.h"

vector<vector<float>> Boundary_detection::read_bin(string filename) {
    vector<vector<float>> pointcloud;
    int32_t num = 1000000;
    float* data = (float*)malloc(num*sizeof(float));
    float *px = data, *py = data+1, *pz = data+2, *pi = data+3, *pr = data+4;

    FILE *stream = fopen(filename.c_str(), "rb");
    num = fread(data, sizeof(float), num, stream) / 5;
    cout << num << endl;
    for (int32_t i = 0; i < num; i++) {
        float dist = std::sqrt((*px)*(*px) + (*py)*(*py) + (*pz)*(*pz));
        float theta = std::atan2(*py, *px) * 180.0f / PI;
        // cout << *px << " " << *py << " " << *pz << " " << *pi << " " << *pr << " " << theta << endl;
        if (dist > 0.9f) pointcloud.push_back({*px, *py, *pz, *pi, *pr, dist, theta});
        px += 5, py += 5, pz += 5, pi += 5, pr += 5;
    }
    for (int i = 0; i < 10; i++) 
        cout << pointcloud[i][0] << " " << pointcloud[i][1] << " " << pointcloud[i][2] << " " << pointcloud[i][3] << " " << pointcloud[i][4] << " " << pointcloud[i][6] << endl;
    for (int i = pointcloud.size()-10; i < pointcloud.size(); i++) 
        cout << pointcloud[i][0] << " " << pointcloud[i][1] << " " << pointcloud[i][2] << " " << pointcloud[i][3] << " " << pointcloud[i][4] << " " << pointcloud[i][6] << endl;
    fclose(stream);
    return pointcloud;
}

void Boundary_detection::rotate_and_translate() {
    if (this->pointcloud.empty()) return;
    // rotation matrix
    // [cos(theta),   0, sin(theta)]
    // [0,            1,          0]
    // [-sin(theta),  0, cos(theta)]
    float theta = this->tilted_angle * PI / 180.0f;
    cout << std::cos(theta) << " " << 0.0f << " " << std::sin(theta) << "\n";
    cout << 0.0f << " " << 1.0f << " " << 0.0f << "\n";
    cout << -std::sin(theta) << " " << 0.0f << " " << std::cos(theta) << "\n";
    for (auto& point : this->pointcloud) {
        int x = point[0] * std::cos(theta) + point[2] * std::sin(theta);
        int y = y;
        int z = point[0] * (-std::sin(theta)) + point[2] * std::cos(theta) + this->sensor_height;
        point[0] = x;
        point[1] = y;
        point[2] = z;
    }
}

void Boundary_detection::max_height_filter(float max_height) {
    auto iter = this->pointcloud.begin(); 
    while (iter != this->pointcloud.end()) {
        if ((*iter)[2] > max_height) iter = this->pointcloud.erase(iter);
        else iter++;
    }
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
    vector<vector<float>> pointcloud_copy(this->pointcloud.begin(), this->pointcloud.end());
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
        [](const vector<float>& p1, const vector<float>& p2){
            if (p1[4] == p2[4]) return p1[6] > p2[6];
            else return p1[4] < p2[4];});
}

void Boundary_detection::pointcloud_preprocessing() {
    rotate_and_translate();
    max_height_filter(this->sensor_height);
}

vector<float> Boundary_detection::get_dist_to_origin() {
    vector<float> dist(this->pointcloud.size());
    for (int i = 0; i < dist.size(); i++) 
        dist[i] = std::sqrt(pointcloud[i][0]*pointcloud[i][0] + pointcloud[i][1]*pointcloud[i][1] + pointcloud[i][2]*pointcloud[i][2]); 
    return dist;
}

float Boundary_detection::dist_between(const vector<float>& p1, const vector<float>& p2) {
    return std::sqrt((p2[0]-p1[0])*(p2[0]-p1[0]) + (p2[1]-p1[1])*(p2[1]-p1[1]) + (p2[2]-p1[2])*(p2[2]-p1[2]));
}

vector<bool> Boundary_detection::continuous_filter(const vector<vector<float>>& pointcloud, const vector<float>& dist_to_origin) {
    int n = pointcloud.size();
    vector<bool> is_continuous(n, true);
    vector<float> thres(dist_to_origin.begin(), dist_to_origin.end());
    for (auto& t : thres) t *= THETA_R * 7;
    for (int i = 0; i < n-1; i++) {
        if (dist_between(pointcloud[i], pointcloud[i+1]) > thres[i]) {
            is_continuous[i] = false;
            is_continuous[i+1] = false;
        }
    }
    return is_continuous;
}

float Boundary_detection::get_angle(const vector<float>& v1, const vector<float>& v2) {
    float angle;
    float dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    float mag_1 = std::sqrt(v1[0]*v1[0] + v1[1]*v1[1] + v1[2]*v1[2]); 
    float mag_2 = std::sqrt(v2[0]*v2[0] + v2[1]*v2[1] + v2[2]*v2[2]); 
    return std::acos(dot_product / (mag_1 * mag_2));
}

vector<float> Boundary_detection::direction_change_filter(int scan_id, int k, float angle_thres /* =150.0f */) {
    int n = this->ranges[scan_id][1] - this->ranges[scan_id][0];
    int st = this->ranges[scan_id][0];
    vector<float> angles(n, 180.0f);
    if (n - (2 * k) < 0) return angles; 
    
    vector<vector<float>> direction_vecs_right(n, vector<float>(3, 0.0f));
    vector<vector<float>> direction_vecs_left(n, vector<float>(3, 0.0f));

    // might need optimize 
    // normalize ? 
    for (int i = 0; i < n-k; i++) {
        for (int j = i+1; j < j+k+1; j++) {
            direction_vecs_right[i][0] = this->pointcloud[st+j][0] - this->pointcloud[st+i][0]; 
            direction_vecs_right[i][1] = this->pointcloud[st+j][1] - this->pointcloud[st+i][1]; 
            direction_vecs_right[i][2] = this->pointcloud[st+j][2] - this->pointcloud[st+i][2]; 
        }
    }
    for (int i = n-1; i >= k; i--) {
        for (int j = i-1; j >= j-k-1; j--) {
            direction_vecs_left[i][0] = this->pointcloud[st+j][0] - this->pointcloud[st+i][0]; 
            direction_vecs_left[i][1] = this->pointcloud[st+j][1] - this->pointcloud[st+i][1]; 
            direction_vecs_left[i][2] = this->pointcloud[st+j][2] - this->pointcloud[st+i][2]; 
        }
    }
    for (int i = k; i < n-k; i++) {
        angles[i] = get_angle(direction_vecs_left[i], direction_vecs_right[i]);
    } 
    return angles;
}

vector<bool> Boundary_detection::local_min_of_direction_change(int scan_id) {

    return {};
}

vector<int> Boundary_detection::elevation_filter(int scan_id) {
    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
    int n = ed - st;
    vector<int> is_elevating(n, 0);
    vector<float> z_diff(n, 0.0f); 
    
    if (scan_id % 2 == 0) { // left scan
        for (int i = ed-2; i >= st; i--) {
            z_diff[i-st] = this->pointcloud[i][2] - this->pointcloud[i+1][2];
            // cout << this->pointcloud[i][2] << " " << this->pointcloud[i+1][2] << " " << z_diff[i] << "\n";
        }
    }
    else {
        for (int i = st+1; i < ed; i++) {
            z_diff[i-st] = this->pointcloud[i][2] - this->pointcloud[i-1][2];
            // cout << this->pointcloud[i][2] << " " << this->pointcloud[i-1][2] << " " << z_diff[i] << "\n";
        }
    }
    for (int i = 0; i < n; i++) {
        if (z_diff[i] > this->dist_to_origin[st+i] * THETA_R) is_elevating[i] = 1;
    }
    return is_elevating;
}

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

void Boundary_detection::edge_filter_from_elevation(int scan_id, const vector<bool>& elevation, vector<bool>& edge_start, vector<bool>& edge_end) {
    int st = this->ranges[scan_id][0], ed = this->ranges[scan_id][1];
    int n = ed - st;
    int k = 7;
    vector<int> f_start, f_end;
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
        if (edge_start_cnt[i+k] >= 2) edge_start[i] = true;
        if (edge_end_cnt[i+k] >= 6) edge_end[i] = true;
    }
}

vector<bool> Boundary_detection::find_boundary_from_half_scan(int scan_id, int k) {

}

vector<bool> Boundary_detection::run_detection() {
    cout << "Run boundary detection...\n";
    auto elevation = elevation_filter(31);
    
    return {};
}


void Boundary_detection::print_pointcloud(const vector<vector<float>>& pointcloud) {
    vector<int> count(16, 0);
    int cnt = 0;
    for (auto& p : pointcloud) {
        // if (p[4] == 0.0f)
        // if (cnt++ < 581)
            cout << p[0] << " " <<  p[1] << " " << p[2] << " " << p[3] << " " << p[4] << " " << p[6] << endl; 
        count[(int)p[4]]++; 
    }
    for (auto& c : count) cout << c << " ";
    cout << endl;
}

vector<vector<float>>& Boundary_detection::get_pointcloud() {
    return this->pointcloud; 
}

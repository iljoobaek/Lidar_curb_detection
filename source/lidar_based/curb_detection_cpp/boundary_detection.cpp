#include "boundary_detection.h"

vector<vector<float>> Boundary_detection::read_bin(string filename) {
    vector<vector<float>> pointcloud;
    int32_t num = 1000000;
    float* data = (float*)malloc(num*sizeof(float));
    float *px = data, *py = data+1, *pz = data+2, *pi = data+3, *pr = data+4;

    FILE *stream = fopen("0000000001.bin", "rb");
    num = fread(data, sizeof(float), num, stream) / 5;
    for (int32_t i = 0; i < num; i++) {
        float dist = sqrt((*px)*(*px) + (*py)*(*py) + (*pz)*(*pz));
        if (dist > 0.) pointcloud.push_back({*px/100, *py/100, *pz/100, *pi, *pr});
        px += 5, py += 5, pz += 5, pi += 5, pr += 5;
    }
    fclose(stream);
    return pointcloud;
}

vector<float> Boundary_detection::get_dist_to_origin(const vector<vector<float>>& pointcloud) {
    vector<float> dist(pointcloud.size());
    for (int i = 0; i < dist.size(); i++) 
        dist[i] = sqrt(pointcloud[i][0]*pointcloud[i][0] + pointcloud[i][1]*pointcloud[i][1] + pointcloud[i][2]*pointcloud[i][2]); 
    return dist;
}

float Boundary_detection::dist_between(const vector<float>& p1, const vector<float>& p2) {
    return sqrt((p2[0]-p1[0])*(p2[0]-p1[0]) + (p2[1]-p1[1])*(p2[1]-p1[1]) + (p2[2]-p1[2])*(p2[2]-p1[2]));
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

void Boundary_detection::print_pointcloud(vector<vector<float>>& pointcloud) {
    vector<int> count(16, 0);
    for (auto& p : pointcloud) {
        cout << p[0] << " " <<  p[1] << " " << p[2] << " " << p[3] << " " << p[4] << endl; 
        count[(int)p[4]]++; 
    }
    for (auto& c : count) cout << c << " ";
    cout << endl;
}

vector<vector<float>>& Boundary_detection::get_pointcloud() {
    return this->pointcloud; 
}

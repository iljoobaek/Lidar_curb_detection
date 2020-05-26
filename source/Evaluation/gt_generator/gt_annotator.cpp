// Modified from pcl library example: "http://pointclouds.org/documentation/tutorials/pcl_visualizer.php"
/* \author Geoffrey Biggs */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <thread>
#include <cmath>
#include <unordered_set>

#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>
#include <pcl/visualization/point_picking_event.h> // for 3D point picking event 

#include <Eigen/Dense>
#define PI 3.14159265

using namespace Eigen;
using namespace std::chrono_literals;

std::vector<std::string> labeled;
std::vector<std::vector<float>> selected_points;
std::vector<int> selected_points_indices;
std::unordered_set<std::string> selected_points_set;

std::string filename_left, filename_right;
std::ofstream file;
bool clicked(false), clear(false);

void rotate_and_translate(std::vector<float> &point, float theta=16.0f, float sensor_height=1.125f)
{
    // rotation matrix along x
    // [1,           0,           0]
    // [0,  cos(theta), -sin(theta)]
    // [0,  sin(theta),  cos(theta)]

    // rotation matrix along y
    // [cos(theta),   0, sin(theta)]
    // [0,            1,          0]
    // [-sin(theta),  0, cos(theta)]
    
    // cout << "[ "<< std::cos(theta) << " " << 0.0f << " " << std::sin(theta) << "\n";
    // cout << 0.0f << " " << 1.0f << " " << 0.0f << "\n";
    // cout << -std::sin(theta) << " " << 0.0f << " " << std::cos(theta) << " ]"<< "\n";

    //theta = theta / 180.0f * PI;
    //float x = point[0] * std::cos(theta) + point[2] * std::sin(theta);
    //float z = point[0] * (-std::sin(theta)) + point[2] * std::cos(theta) + sensor_height;
    //point[0] = x;
    //point[2] = z;
    
    // Rotate along z axis to match the coordinates from pcap / velodyne capture
    float xx = point[0] * std::cos(PI / 2) + point[1] * (-std::sin(PI / 2));
    float yy = point[0] * std::sin(PI / 2) + point[1] * std::cos(PI / 2);
    point[0] = xx;
    point[1] = yy;
}

std::vector<float> getThirdOrderPolynomials() 
{
    std::vector<float> boundaryCoeffs;

    // Calculate third order polynomials by linear least square 
    MatrixXf A(selected_points.size(), 4);
    VectorXf b(selected_points.size());
    for (int i = 0; i < selected_points.size(); i++) 
    {
        A(i, 0) = std::pow(selected_points[i][1], 3);
        A(i, 1) = std::pow(selected_points[i][1], 2);
        A(i, 2) = std::pow(selected_points[i][1], 1);
        A(i, 3) = 1.0f;
    }
    for (int i = 0; i < selected_points.size(); i++) 
    {
        b(i) = selected_points[i][0];
    }
    
    // VectorXf solution = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
    VectorXf solution = A.colPivHouseholderQr().solve(b);
    std::cout << "The least-squares solution is:\n" << solution << std::endl;
    for (int i = 0; i < 4; i++)
    {
        boundaryCoeffs.push_back(solution(i));
    }
    return boundaryCoeffs;
}

void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
        void* viewer_void)
{
    pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
    if (event.getKeySym () == "l" && event.keyDown ())
    {
        std::vector<float> boundaryCoeffs = getThirdOrderPolynomials();
        std::stringstream ss;
        assert(boundaryCoeffs.size() == 4);
        ss << boundaryCoeffs[0] << " " << boundaryCoeffs[1] << " " << boundaryCoeffs[2] << " " << boundaryCoeffs[3] << "\n";    

        file.open(filename_left);
        file << ss.str();
        for (auto &point : labeled) 
        {
            file << point;
        }
        file.close();
        std::cout << "Save left boundary points to txt" << std::endl;
    }
    else if (event.getKeySym () == "r" && event.keyDown ())
    {
        std::vector<float> boundaryCoeffs = getThirdOrderPolynomials();
        std::stringstream ss;
        assert(boundaryCoeffs.size() == 4);
        ss << boundaryCoeffs[0] << " " << boundaryCoeffs[1] << " " << boundaryCoeffs[2] << " " << boundaryCoeffs[3] << "\n";    
        
        file.open(filename_right);
        file << ss.str();
        for (auto &point : labeled) 
        {
            file << point;
        }
        file.close();
        std::cout << "Save right boundary points to txt" << std::endl;
    }
    else if (event.getKeySym () == "k" && event.keyDown ())
    {
        labeled.clear();
        selected_points.clear();
        selected_points_indices.clear();
        selected_points_set.clear();
        std::cout << "Clear selected points" << std::endl;
        clear = true;
    }
}

void pointPickingEventOccurred(const pcl::visualization::PointPickingEvent& event, 
        void* viewer_void)
{
    pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
    float x, y, z;
    if (event.getPointIndex() == -1)
    {
        return;
    }
    event.getPoint(x, y, z);
    std::stringstream ss;
    ss << x << " " << y << " " << z << "\n";
    std::string newPoint = ss.str();    
    if (selected_points_set.find(newPoint) == selected_points_set.end())
    {
        selected_points_set.insert(newPoint);
        labeled.push_back(ss.str());
        selected_points.push_back({x, y, z});
        selected_points_indices.push_back(event.getPointIndex());
        std::cout << "Point with index " << event.getPointIndex()  << " at ( " << x << ", " << y << ", " << z << " )" << std::endl;
    }
    else
    {
        std::cout << "Point with index" << event.getPointIndex() << " already selected\n";
    }
    clicked = true;
}

pcl::visualization::PCLVisualizer::Ptr interactionCustomizationVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud, std::ofstream &file)
{
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem (1.0);

    viewer->registerKeyboardCallback (keyboardEventOccurred, (void*)viewer.get ());
    //viewer->registerMouseCallback (mouseEventOccurred, (void*)viewer.get ());
    viewer->registerPointPickingCallback(pointPickingEventOccurred, (void*)&viewer);

    return (viewer);
}

pcl::PointCloud<pcl::PointXYZRGB>::Ptr create_cloud_from_binary (const std::string &filename, bool isKitti)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    if (isKitti)
    {
        int32_t num = 1000000;
        float *data = (float *)malloc(num * sizeof(float));
        float *px = data, *py = data + 1, *pz = data + 2, *pi = data + 3;

        FILE *stream = fopen(filename.c_str(), "rb");
        num = fread(data, sizeof(float), num, stream) / 4;
        for (int32_t i = 0; i < num; i++)
        {
            std::vector<float> point({*px, *py, *pz});
            rotate_and_translate(point);
            
            pcl::PointXYZRGB rgb_point;
            rgb_point.x = point[0];
            rgb_point.y = point[1];
            rgb_point.z = point[2];
            uint8_t r(255), g(255), b(255);
            uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
            rgb_point.rgb = *reinterpret_cast<float*>(&rgb);
            point_cloud_ptr->points.push_back (rgb_point); 
            px += 4, py += 4, pz += 4, pi += 4;
        }
        fclose(stream);
    }
    else
    {
        int32_t num = 1000000;
        float *data = (float *)malloc(num * sizeof(float));
        float *px = data, *py = data + 1, *pz = data + 2, *pi = data + 3, *pr = data + 4;

        FILE *stream = fopen(filename.c_str(), "rb");
        num = fread(data, sizeof(float), num, stream) / 5;
        for (int32_t i = 0; i < num; i++)
        {
            std::vector<float> point({*px, *py, *pz});
            rotate_and_translate(point);
            
            pcl::PointXYZRGB rgb_point;
            rgb_point.x = point[0];
            rgb_point.y = point[1];
            rgb_point.z = point[2];
            uint8_t r(255), g(255), b(255);
            uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
            rgb_point.rgb = *reinterpret_cast<float*>(&rgb);
            point_cloud_ptr->points.push_back (rgb_point); 
            px += 5, py += 5, pz += 5, pi += 5, pr += 5;
        }
        fclose(stream);
    }

    //int32_t num = 1000000;
    //float *data = (float *)malloc(num * sizeof(float));
    //float *px = data, *py = data + 1, *pz = data + 2, *pi = data + 3, *pr = data + 4;

    //FILE *stream = fopen(filename.c_str(), "rb");
    //num = fread(data, sizeof(float), num, stream) / 5;
    //for (int32_t i = 0; i < num; i++)
    //{
    //    std::vector<float> point({*px, *py, *pz});
    //    rotate_and_translate(point);
    //    
    //    pcl::PointXYZRGB rgb_point;
    //    rgb_point.x = point[0];
    //    rgb_point.y = point[1];
    //    rgb_point.z = point[2];
    //    uint8_t r(255), g(255), b(255);
    //    uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
    //    rgb_point.rgb = *reinterpret_cast<float*>(&rgb);
    //    point_cloud_ptr->points.push_back (rgb_point); 
    //    px += 5, py += 5, pz += 5, pi += 5, pr += 5;
    //}
    //fclose(stream);
    return point_cloud_ptr;
}

int main (int argc, char** argv)
{
    bool isKitti(false);
    std::string fn;
    if (argc == 2)
    {
        fn = argv[1];
    }
    else if (argc == 3)
    {
        fn = argv[1];
        std::string source = argv[2];
        if (source == "kitti")
        {
            isKitti = true;
        }
        else
        {
            std::cerr << "Second argument invalid!\n";
        }
    }
    else
    {
        std::cerr << "Invalid number of arguments!\n";
        return -1; 
    }

    int st = fn.find('/');
    filename_left = "evaluation_result/gt_" + fn.substr(st+1, 10) + "_l.txt";
    filename_right = "evaluation_result/gt_" + fn.substr(st+1, 10) + "_r.txt";

    // Load point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    auto point_cloud_ptr = create_cloud_from_binary (fn, isKitti);

    pcl::visualization::PCLVisualizer::Ptr viewer;
    viewer = interactionCustomizationVis(point_cloud_ptr, file);
    
    // Main loop
    while (!viewer->wasStopped ())
    {
        if (clicked) 
        {
            uint8_t r(0), g(255), b(0);
            uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
            point_cloud_ptr->points[selected_points_indices.back()].rgb = *reinterpret_cast<float*>(&rgb);
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_handle(point_cloud_ptr);
            viewer->updatePointCloud<pcl::PointXYZRGB>(point_cloud_ptr, rgb_handle, "sample cloud"); 
            clicked = false;
        }
        if (clear) 
        {
            uint8_t r(255), g(255), b(255);
            uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
            for (auto &point: point_cloud_ptr->points) 
            {
                point.rgb = *reinterpret_cast<float*>(&rgb);
            }
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb_handle(point_cloud_ptr);
            viewer->updatePointCloud<pcl::PointXYZRGB>(point_cloud_ptr, rgb_handle, "sample cloud"); 
            std::cout << "Selected point buffer cleared\n";
            clear = false;
        }
        viewer->spinOnce (100);
        std::this_thread::sleep_for(100ms);
    }
}

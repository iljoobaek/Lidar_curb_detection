// Modified from pcl library example: "http://pointclouds.org/documentation/tutorials/pcl_visualizer.php"
/* \author Geoffrey Biggs */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <thread>

#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include <pcl/visualization/point_picking_event.h> // for 3D point picking event 

using namespace std::chrono_literals;

std::vector<std::string> labeled;
std::vector<std::vector<float>> selected_points;
std::vector<int> selected_points_indices;

std::string filename_left, filename_right;
std::ofstream file;
bool clicked(false), clear(false);

void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
        void* viewer_void)
{
    pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
    if (event.getKeySym () == "l" && event.keyDown ())
    {
        file.open(filename_left);
        for (auto &point : labeled) 
        {
            file << point;
        }
        file.close();
        std::cout << "Save left boundary points to txt" << std::endl;
    }
    else if (event.getKeySym () == "r" && event.keyDown ())
    {
        file.open(filename_right);
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
    labeled.push_back(ss.str());
    selected_points.push_back({x, y, z});
    selected_points_indices.push_back(event.getPointIndex());
    clicked = true;
    std::cout << "Point with index " << event.getPointIndex()  << " at ( " << x << ", " << y << ", " << z << " )" << std::endl;
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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr create_cloud_from_binary (const std::string &filename)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    int32_t num = 1000000;
    float *data = (float *)malloc(num * sizeof(float));
    float *px = data, *py = data + 1, *pz = data + 2, *pi = data + 3, *pr = data + 4;

    FILE *stream = fopen(filename.c_str(), "rb");
    num = fread(data, sizeof(float), num, stream) / 5;
    for (int32_t i = 0; i < num; i++)
    {
        pcl::PointXYZRGB rgb_point;
        rgb_point.x = *px;
        rgb_point.y = *py;
        rgb_point.z = *pz;
        uint8_t r(255), g(255), b(255);
        uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
        rgb_point.rgb = *reinterpret_cast<float*>(&rgb);
        point_cloud_ptr->points.push_back (rgb_point); 
        px += 5, py += 5, pz += 5, pi += 5, pr += 5;
    }
    fclose(stream);
    return point_cloud_ptr;
}

int main (int argc, char** argv)
{
    std::string fn;
    if (argc < 2)
    {
        fn = "evaluation_data/0000000000.bin";
    }
    else if (argc == 2)
    {
        fn = argv[1];
    }
    else
    {
        std::cout << "Invalid number of arguments!\n";
        return -1; 
    }

    int st = fn.find('/');
    filename_left = "evaluation_result/gt_" + fn.substr(st+1, 10) + "_l.txt";
    filename_right = "evaluation_result/gt_" + fn.substr(st+1, 10) + "_r.txt";

    // Load point cloud
    pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    auto point_cloud_ptr = create_cloud_from_binary (fn);

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

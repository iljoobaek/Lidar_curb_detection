// Modified from pcl library example: "http://pointclouds.org/documentation/tutorials/pcl_visualizer.php"
/* \author Geoffrey Biggs */

#include <iostream>
#include <vector>
#include <string>
#include <thread>

#include <pcl/common/common_headers.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/console/parse.h>

#include <pcl/visualization/point_picking_event.h> // for 3D point picking event 

using namespace std::chrono_literals;

// --------------
// -----Help-----
// --------------
void printUsage (const char* progName)
{
    std::cout << "\n\nUsage: "<<progName<<" [options]\n\n"
        << "Options:\n"
        << "-------------------------------------------\n"
        << "-h           this help\n"
        << "-s           Simple visualisation example\n"
        << "-r           RGB colour visualisation example\n"
        << "-c           Custom colour visualisation example\n"
        << "-i           Interaction Customization example\n"
        << "\n\n";
}


pcl::visualization::PCLVisualizer::Ptr simpleVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    return (viewer);
}


pcl::visualization::PCLVisualizer::Ptr rgbVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb(cloud);
    viewer->addPointCloud<pcl::PointXYZRGB> (cloud, rgb, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    return (viewer);
}


pcl::visualization::PCLVisualizer::Ptr customColourVis (pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
    // --------------------------------------------
    // -----Open 3D viewer and add point cloud-----
    // --------------------------------------------
    pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
    viewer->setBackgroundColor (0, 0, 0);
    pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 255, 0);
    viewer->addPointCloud<pcl::PointXYZ> (cloud, single_color, "sample cloud");
    viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "sample cloud");
    viewer->addCoordinateSystem (1.0);
    viewer->initCameraParameters ();
    return (viewer);
}

unsigned int text_id = 0;
void keyboardEventOccurred (const pcl::visualization::KeyboardEvent &event,
        void* viewer_void)
{
    pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
    if (event.getKeySym () == "r" && event.keyDown ())
    {
        std::cout << "r was pressed => removing all text" << std::endl;

        char str[512];
        for (unsigned int i = 0; i < text_id; ++i)
        {
            sprintf (str, "text#%03d", i);
            viewer->removeShape (str);
        }
        text_id = 0;
    }
}

void mouseEventOccurred (const pcl::visualization::MouseEvent &event,
        void* viewer_void)
{
    pcl::visualization::PCLVisualizer *viewer = static_cast<pcl::visualization::PCLVisualizer *> (viewer_void);
    if (event.getButton () == pcl::visualization::MouseEvent::LeftButton &&
            event.getType () == pcl::visualization::MouseEvent::MouseButtonRelease)
    {
        std::cout << "Left mouse button released at position (" << event.getX () << ", " << event.getY () << ")" << std::endl;

        char str[512];
        sprintf (str, "text#%03d", text_id ++);
        viewer->addText ("clicked here", event.getX (), event.getY (), str);
    }
}

void pointPickingEventOccurred(const pcl::visualization::PointPickingEvent& event, 
        void* viewer_void)
{
    float x, y, z;
    if (event.getPointIndex() == -1)
    {
        return;
    }
    event.getPoint(x, y, z);
    std::cout << "Point coordinate ( " << x << ", " << y << ", " << z << ")" << std::endl;
    //selectedPoints.push_back(pcl::PointXYZ(x, y, z));
}

pcl::visualization::PCLVisualizer::Ptr interactionCustomizationVis (pcl::PointCloud<pcl::PointXYZRGB>::ConstPtr cloud)
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

pcl::PointCloud<pcl::PointXYZRGB>::Ptr create_cloud_from_binary (const std::string& filename)
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    
    int32_t num = 1000000;
    float *data = (float *)malloc(num * sizeof(float));
    float *px = data, *py = data + 1, *pz = data + 2, *pi = data + 3, *pr = data + 4;

    FILE *stream = fopen(filename.c_str(), "rb");
    num = fread(data, sizeof(float), num, stream) / 5;
    for (int32_t i = 0; i < num; i++)
    {
//        pcl::PointXYZ basic_point;
//        basic_point.x = *px;
//        basic_point.y = *py;
//        basic_point.z = *pz;
        
        pcl::PointXYZRGB rgb_point;
        rgb_point.x = *px;
        rgb_point.y = *py;
        rgb_point.z = *pz;
        uint8_t r(255), g(15), b(15);
        uint32_t rgb = (static_cast<uint32_t>(r) << 16 | static_cast<uint32_t>(g) << 8 | static_cast<uint32_t>(b));
        rgb_point.rgb = *reinterpret_cast<float*>(&rgb);
        point_cloud_ptr->points.push_back (rgb_point); 
        px += 5, py += 5, pz += 5, pi += 5, pr += 5;
    }
    fclose(stream);
    return point_cloud_ptr;
}

// --------------
// -----Main-----
// --------------
int main (int argc, char** argv)
{
    // --------------------------------------
    // -----Parse Command Line Arguments-----
    // --------------------------------------
    if (pcl::console::find_argument (argc, argv, "-h") >= 0)
    {
        printUsage (argv[0]);
        return 0;
    }
    bool simple(false), rgb(false), custom_c(false), normals(false),
         shapes(false), viewports(false), interaction_customization(false);
    if (pcl::console::find_argument (argc, argv, "-s") >= 0)
    {
        simple = true;
        std::cout << "Simple visualisation example\n";
    }
    else if (pcl::console::find_argument (argc, argv, "-c") >= 0)
    {
        custom_c = true;
        std::cout << "Custom colour visualisation example\n";
    }
    else if (pcl::console::find_argument (argc, argv, "-r") >= 0)
    {
        rgb = true;
        std::cout << "RGB colour visualisation example\n";
    }
    else if (pcl::console::find_argument (argc, argv, "-i") >= 0)
    {
        interaction_customization = true;
        std::cout << "Interaction Customization example\n";
    }
    else
    {
        printUsage (argv[0]);
        return 0;
    }

    //    // ------------------------------------
    //    // -----Create example point cloud-----
    //    // ------------------------------------
    //    pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    //    pcl::PointCloud<pcl::PointXYZRGB>::Ptr point_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
    //

    pcl::PointCloud<pcl::PointXYZ>::Ptr basic_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
    auto point_cloud_ptr = create_cloud_from_binary ("0000000000.bin");

    pcl::visualization::PCLVisualizer::Ptr viewer;
    if (simple)
    {
        viewer = simpleVis(basic_cloud_ptr);
    }
    else if (rgb)
    {
        viewer = rgbVis(point_cloud_ptr);
    }
    else if (custom_c)
    {
        viewer = customColourVis(basic_cloud_ptr);
    }
    else if (interaction_customization)
    {
        viewer = interactionCustomizationVis(point_cloud_ptr);
    }

    //--------------------
    // -----Main loop-----
    //--------------------
    while (!viewer->wasStopped ())
    {
        viewer->spinOnce (100);
        std::this_thread::sleep_for(100ms);
    }
}

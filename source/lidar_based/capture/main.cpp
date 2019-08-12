#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <opencv2/opencv.hpp>
#include <opencv2/viz.hpp>
#include <thread>
#include <iomanip>
// Include VelodyneCapture Header
#include "VelodyneCapture.h"

void laser_to_cartesian(std::vector<velodyne::Laser>& lasers, std::vector<cv::Vec3f>& buffer, std::vector<float>& data) {
    buffer.resize(lasers.size());
    data.resize(lasers.size()*5);
    int idx = 0;
    for (int i = 0; i < lasers.size(); i++) {
        const double distance = static_cast<double>( lasers[i].distance );
        const double azimuth  = lasers[i].azimuth  * CV_PI / 180.0;
        const double vertical = lasers[i].vertical * CV_PI / 180.0;

        float x = static_cast<float>( ( distance * std::cos( vertical ) ) * std::sin( azimuth ) );
        float y = static_cast<float>( ( distance * std::cos( vertical ) ) * std::cos( azimuth ) );
        float z = static_cast<float>( ( distance * std::sin( vertical ) ) );

        if( x == 0.0f && y == 0.0f && z == 0.0f ) continue;

        float intensity = static_cast<float>(lasers[i].intensity);
        float ring = static_cast<float>(lasers[i].id);

        data[idx*5] = x;
        data[idx*5+1] = y;
        data[idx*5+2] = z;
        data[idx*5+3] = intensity;
        data[idx*5+4] = ring;

        buffer[idx] = cv::Vec3f( x, y, z );
        idx++;
    }
    buffer.resize(idx);
    data.resize(idx*5);
}

void write_bin(const std::vector<float>& data, std::string dst, int idx) {
    std::stringstream ss;
    ss << std::setw(10) << std::setfill('0') << idx;
    std::string fn = dst + ss.str() + ".bin";
    std::cout << fn << std::endl;
    auto stream = fopen(fn.c_str(), "wb");
    fwrite(&data[0], sizeof(float), data.size(), stream);
    fclose(stream);
}

int main( int argc, char* argv[] )
{
    /*
    // Open VelodyneCapture that retrieve from Sensor
    const boost::asio::ip::address address = boost::asio::ip::address::from_string( "192.168.1.201" );
    const unsigned short port = 2368;
    velodyne::VLP16Capture capture( address, port );
    //velodyne::HDL32ECapture capture( address, port );
    */

    // Open VelodyneCapture that retrieve from PCAP
    const std::string filename = "kesselRun.pcap";
    velodyne::VLP16Capture capture( filename );
    //velodyne::HDL32ECapture capture( filename ); 
    const std::string dst = "test/";

    if( !capture.isOpen() ){
        std::cerr << "Can't open VelodyneCapture." << std::endl;
        return -1;
    }

    // Create Viewer
    cv::viz::Viz3d viewer( "Velodyne" );

    // Register Keyboard Callback
    viewer.registerKeyboardCallback(
            []( const cv::viz::KeyboardEvent& event, void* cookie ){
            // Close Viewer
            if( event.code == 'q' && event.action == cv::viz::KeyboardEvent::Action::KEY_DOWN ){
            static_cast<cv::viz::Viz3d*>( cookie )->close();
            }
            }
            , &viewer
            );

    int idx = 0; 
    while( capture.isRun() && !viewer.wasStopped() ){
        auto now = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();

        std::vector<velodyne::Laser> lasers;
        std::vector<cv::Vec3f> buffer;
        std::vector<float> data;

        // Capture One Rotation Data
        capture >> lasers;
        if( lasers.empty() ){
            continue;
        }
        laser_to_cartesian(lasers, buffer, data);
        // write to binary
        write_bin(data, dst, idx);        

        auto t = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - now;
        std::cout << "Take " << t << " ms ";

        // Create Widget
        cv::Mat cloudMat = cv::Mat( static_cast<int>( buffer.size() ), 1, CV_32FC3, &buffer[0] );
        cv::viz::WCloud cloud( cloudMat );

        // Show Point Cloud
        viewer.showWidget("Coordinate Widget", cv::viz::WCoordinateSystem(100));
        viewer.showWidget( "Cloud", cloud );
        viewer.spinOnce();
        //getchar();
        auto t2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count() - now;
        idx++;
    }
    // Close All Viewers
    cv::viz::unregisterAllWindows();

    return 0;
}

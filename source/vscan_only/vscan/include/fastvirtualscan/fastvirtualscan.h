#ifndef FASTVIRTUALSCAN_H
#define FASTVIRTUALSCAN_H

#include<QVector>
#include<QtAlgorithms>
#include<QtGlobal>
#include<sensor_msgs/PointCloud2.h>

#include<vector>

#ifndef CV2_H
#define CV2_H
#include<opencv2/opencv.hpp>
#include<opencv2/viz.hpp>
#endif

struct SimpleVirtualScan
{
    int rotid;
    double rotlength;
    double rotheight;
    double length;
    double height;
};

class FastVirtualScan
{
public:
    // sensor_msgs::PointCloud2ConstPtr velodyne;
public:
    int beamnum;
    double step;
    double minfloor;
    double maxceiling;
    double rotation;
    double minrange;
    SimpleVirtualScan *svs;
    SimpleVirtualScan *svsback;
    QVector<double> minheights;
    QVector<double> maxheights;
public:
    FastVirtualScan(int beamNum, double heightStep, double minFloor, double maxCeiling);
    virtual ~FastVirtualScan();
public:
    void calculateVirtualScans(const std::vector<cv::Vec3f> &pointcloud, int beamNum, double heightStep, double minFloor, double maxCeiling, double obstacleMinHeight=1, double maxBackDistance=1, double beamRotation=0, double minRange=0);
    void getVirtualScan(double thetaminheight, double thetamaxheight, double maxFloor, double minCeiling, double passHeight, QVector<double> & virtualScan);
    
    void printsvs();
};

#endif // FASTVIRTUALSCAN_H

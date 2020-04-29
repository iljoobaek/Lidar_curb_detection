# Boundary (Curb detection) with virtualscan

To upgrade to the latest version, please do the following:

Add the Ubuntu repository that contains the latest version 2019~U8-1, run the following command in terminal:

echo "deb http://cz.archive.ubuntu.com/ubuntu eoan main universe" | sudo tee -a  /etc/apt/sources.list

Update the repositories, run the following command in terminal:

sudo apt update

Upgrade to the latest version, run the following command in terminal:

sudo apt install libtbb-dev

After this you should have the latest libtbb-dev installed.


## Data path

### To run with our dataset
```cpp
Boundary_detection detection(16, 1.125, "/home/rtml/lidar_radar_fusion_curb_detection/data/", "20191126163620_synced/", frameStart, frameEnd+1, false);
```
The first 16 means 16 scanlines from our data.
The third argument is the root path of the data folder and the fourth argument is the data folder.

### To run with kitti dataset
```cpp
Boundary_detection detection(64, 1.125, "/home/rtml/LiDAR_camera_calibration_work/data/kitti_data/2011_09_26/", "2011_09_26_drive_0013_sync/", frameStart, frameEnd+1, false);
```
Kitti dataset comes with 64 scanlines so the first argument is 64.
The last argument here could be true or false, which means downsample the point cloud or not. 
If true, the 64 scans will be divided by 4, to treat it as the 16 scans data.

### Notice
The current implementation of recognizing whether the data is from kitti or not simply see if the root path
has "kitti" included as below.
```cpp
root_path.find("kitti") == std::string::npos ? false : true;
```
So make sure the root path is something like "/home/data_set/kitti_data/...".

## Run the program from command line 
```
./boundary_detection
```

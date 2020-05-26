#!env python

import sys, os
sys.path.insert(0, '/usr/local/lib/python2.7/dist-packages/')
import pcl

import rospy, time
import rosbag
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, PointField
import sensor_msgs.point_cloud2 as pcl2
import numpy as np 
#print np.version.version
from datetime import datetime
import cv2
from cv_bridge import CvBridge

class RosbagWriter:
    def __init__(self, bag_name, data_path):
        self.bag_name = bag_name
        self.bag = rosbag.Bag(bag_name + '.bag', 'w')
        self.bridge = CvBridge()
        
        self.folder_path = os.path.join(data_path, self.bag_name)
        # velodyne info
        # self.velo_frame_id = 'velo_link'
        self.velo_frame_id = 'velodyne'
        self.velo_topic = '/kitti/velo'
        
        # cam_info 
        self.cam1_frame_id = 'cam_1'
        self.cam1_topic = '/kitti/cam1'
    
    def save_velo_data(self):
        print("Exporting velodyne data")
        # set velodyne data input path
        velo_path = os.path.join(self.folder_path, 'velodyne_points/')
        velo_data_dir = os.path.join(velo_path, 'data')
        print velo_data_dir
        velo_filenames = sorted(os.listdir(velo_data_dir))
        #print velo_filenames
        
        with open(os.path.join(velo_path, 'timestamps.txt')) as f:
            lines = f.readlines()
            velo_datetimes = []
            for line in lines:
                if len(line) == 1:
                    continue
                dt = datetime.strptime(line[:-4], '%Y-%m-%d %H:%M:%S.%f')
                ut_bag = RosbagWriter(sys.argv[1], '../')    

                velo_datetimes.append(dt)
        
        print len(velo_datetimes),  len(velo_filenames)        

        iterable = zip(velo_datetimes, velo_filenames)
        for dt, filename in iterable:
            if dt is None:
                continue

            velo_filename = os.path.join(velo_data_dir, filename)

            # read binary data / read from pcd file ??
            scan = (np.fromfile(velo_filename, dtype=np.float32)).reshape(-1, 4)

            # create header
            header = Header()
            header.frame_id = self.velo_frame_id
            header.stamp = rospy.Time.from_sec(float(datetime.strftime(dt, "%s.%f")))

            # fill pcl msg
            fields = [PointField('x', 0, PointField.FLOAT32, 1),
                      PointField('y', 4, PointField.FLOAT32, 1),
                      PointField('z', 8, PointField.FLOAT32, 1),
                      PointField('i', 12, PointField.FLOAT32, 1)]
            pcl_msg = pcl2.create_cloud(header, fields, scan)

            # self.bag.write(self.velo_topic + '/pointcloud', pcl_msg, t=pcl_msg.header.stamp)
            self.bag.write('/points_raw', pcl_msg, t=pcl_msg.header.stamp)

    def save_camera_data(self, camera):
        print("Exporting camera {}".format(camera))
        #cam_path = os.path.join(self.data_path_cam1, self.bag_name)
        
        camera_pad = '{0:02d}'.format(camera)
        image_dir = os.path.join(self.folder_path, 'image_{}/'.format(camera_pad))
        image_path = os.path.join(image_dir, 'data')
        print image_path
        image_filenames = sorted(os.listdir(image_path))
 
        with open(os.path.join(image_dir, 'timestamps.txt')) as f:
            image_datetimes = map(lambda x: datetime.strptime(x[:-4], '%Y-%m-%d %H:%M:%S.%f'), f.readlines())
        print len(image_datetimes), len(image_filenames)        

        iterable = zip(image_datetimes, image_filenames)
        for dt, filename in iterable:
            image_filename = os.path.join(image_path, filename)
            cv_image = cv2.imread(image_filename)
            # print cv_image.shape
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            encoding = "mono8"
            # print cv_image.shape
            #if camera in (0, 1):
            #    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
            #encoding = "mono8" if camera in (0, 1) else "bgr8"

            image_message = self.bridge.cv2_to_imgmsg(cv_image, encoding=encoding)
            image_message.header.frame_id = self.cam1_frame_id
                
            image_message.header.stamp = rospy.Time.from_sec(float(datetime.strftime(dt, "%s.%f")))
            topic_ext = "/image_raw"
            
            #self.bag.write(self.cam1_topic + topic_ext, image_message, t = image_message.header.stamp)
            self.bag.write(self.cam1_frame_id + topic_ext, image_message, t = image_message.header.stamp)

    def run(self):
        self.save_velo_data()
        self.save_camera_data(1)
        self.bag.close() 


if __name__ == "__main__":
     
    if len(sys.argv) > 1:
        print sys.argv[1]
        # out_bag = RosbagWriter(sys.argv[1], '../')    
        out_bag = RosbagWriter(sys.argv[1], '../../data/data_raw/synced/')    
    else:
        print "No argument passed in"
        out_bag = RosbagWriter('test', ['../'])    
    
    out_bag.run()




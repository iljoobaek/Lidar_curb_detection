import os
import time
from os.path import isfile, join
import sys
import numpy as np
import cv2
import rosbag
import sensor_msgs.point_cloud2 as pc2

def get_pointcloud_from_msg(msg):
    """
    """
    pc_list = list(pc2.read_points(msg))
    return np.array(pc_list, 'float32')

if __name__ == "__main__":
    # topics in the bag file
    topics = ['/camera/image_raw', '/points_raw']
    
    if len(sys.argv) > 1:
        bagname = sys.argv[1]
        bag = rosbag.Bag(bagname)
        topic_camera = bag.read_messages(topics[0])
        topic_lidar = bag.read_messages(topics[1])
        
        directory = 'data/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        idx = 0
        for topic, msg, t in topic_lidar:
            pointcloud = get_pointcloud_from_msg(msg)
            fn = directory + str(idx).zfill(10) + '.bin' 
            f = open(fn, mode='wb')
            np.asarray(pointcloud, dtype=np.float32).tofile(f)
            idx += 1
    else:
        print 'no bagname'



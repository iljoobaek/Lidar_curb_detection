import rosbag
from cv_bridge import CvBridge, CvBridgeError # convert between ROS image messages and openCV image
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2 
import sys, os, errno

# The class RosbagParser read in a rosbag file with two topics [/image_raw, /points_raw]
# and output synced data in [.png, .bin]

class RosbagParser:
    def __init__(self, bagfile, topics):
        self.bag = rosbag.Bag(bagfile)
        self.topic_0 = self.bag.read_messages(topics[0])
        self.topic_1 = self.bag.read_messages(topics[1])
        self.len_0 = self.bag.get_message_count(topics[0])
        self.len_1 = self.bag.get_message_count(topics[1])
        self.bridge = CvBridge()
        self.folder = bagfile.split('.')[0]

    def write_to_image(self, msg, ind):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            fname = self.folder +"/image/data/" + '{:010d}'.format(ind) + ".png"
            if not os.path.exists(os.path.dirname(fname)):
                try:
                    os.makedirs(os.path.dirname(fname))
                except OSError as exc:
                    if exc.errno != errno.EEXIST:
                        raise
            cv2.imwrite(fname, cv_image) 
        except CvBridgeError, e:
            print e

    def write_to_bin(self, msg, ind):
        fname = self.folder + "/velodyne_points/data/" + '{:010d}'.format(ind) + ".bin"
        if not os.path.exists(os.path.dirname(fname)):
            try:
                os.makedirs(os.path.dirname(fname))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        f = file(fname, "wb")
        for p in pc2.read_points(msg):
            arr = np.array(p[0:4],'float32') # x y z intensity 
            arr.tofile(f)
        f.close()

    def sync_data(self):
        # start from topic0
        ind = 0 
        j = 0
        ii = 0
        for topic_0, msg_0, t_0 in self.topic_0:
            ii = ii + 1
            time_0 = msg_0.header.stamp.to_sec()
            
            jj = 0
            k = 0
            self.topic_1 = self.bag.read_messages(topics[1])
            for topic_1, msg_1, t_1 in self.topic_1:
                jj = jj + 1
                k = k + 1
                time_1 = t_1.to_sec()
                if k <= j:
                    continue
                if abs(time_1 - time_0) < 0.1:
                    # write to image
                    self.write_to_image(msg_0, ind)
                    # write to bin
                    self.write_to_bin(msg_1, ind)
                    print "index " + str(ind) + ": i = " + str(ii) + " j = " + str(jj) 
                    j = k
                    ind = ind + 1
                    break
    
    def close_bag(self):
        self.bag.close()

    def read_points(self):
        idx = 0
        for topic_1, msg_1, t_1 in self.topic_1:
            print msg_1.height, msg_1.width
            fname = str(idx) + ".txt"
            with open(fname, 'a') as f:
                array = np.empty((0,5), 'float32')
                for p in pc2.read_points(msg_1):
                    arr = np.array(p[:],'float32') # x y z intensity 
                    array = np.vstack((array, arr))
                    # print arr
                if idx == 300:
                    np.savetxt(f, array)
                    break
if __name__ == "__main__":
    
    # topics in the bag file
    topics = ['/camera/image_raw', '/points_raw']
    print topics
    
    if len(sys.argv) > 1:
        my_parser = RosbagParser(sys.argv[1], topics)
    else:
        my_parser = RosbagParser("out1.bag", topics)
    
    print str(my_parser.bag.get_message_count(topics[0])) + " messages in " + topics[0]
    print str(my_parser.bag.get_message_count(topics[1])) + " messages in " + topics[1]
    
    # my_parser.sync_data()
    # my_parser.close_bag()
    
    my_parser.read_points()


from __future__ import print_function
from rosbag import Bag
from cv_bridge import CvBridge, CvBridgeError
import rospy
import argparse

def parseRosbag(bag):
    t_start = bag.get_start_time() # return float in seconds
    t_end = bag.get_end_time()
    t_split = []
    while t_start < t_end:
        if t_start + 60.0 > t_end:
            t_split.append([t_start, t_end])
        else:
            t_split.append([t_start, t_start+60.0]) 
        t_start += 60.0
    
    fn = bag.filename.split('.')[0] + '_'
    for i in range(len(t_split)):
        fn_split = fn + str(i) + '.bag'
        with Bag(fn_split, 'w') as out_bag:
            t_start = rospy.Time(t_split[i][0])
            t_end = rospy.Time(t_split[i][1])
            t_0 = rospy.Duration(secs=0)
            for topic, msg, t in bag:
                t_d1 = t - t_start
                t_d2 = t_end - t
                if t_d1 < t_0:
                    continue
                if t_d2 < t_0:
                    break
                if topic == 'lidar_front' or topic == '/points_raw':
                    out_bag.write('/points_raw', msg, t)
                elif topic == 'image_raw' or topic == '/image_raw':
                    try:
                        cv_image = CvBridge().imgmsg_to_cv2(msg)
                        converted_msg = CvBridge().cv2_to_imgmsg(cv_image, "bgr8")
                    except CvBridgeError as e:
                        print(e)
                    out_bag.write('/image_raw', converted_msg, t)
            out_bag.close()
    return 

def toRGB(bag):
    fn = bag.filename.split('.')[0] + '_rgb.bag'
    cnt = 0
    with Bag(fn, 'w') as out_bag:
        for topic, msg, t in bag:
            if topic == 'lidar_front' or topic == '/points_raw':
                out_bag.write('/points_raw', msg, t)
            elif topic == 'image_raw' or topic == '/image_raw':
                try:
                    cv_image = CvBridge().imgmsg_to_cv2(msg)
                    converted_msg = CvBridge().cv2_to_imgmsg(cv_image, "bgr8")
                except CvBridgeError as e:
                    print(e)
                out_bag.write('/image_raw', converted_msg, t)
        out_bag.close()

parser = argparse.ArgumentParser()
parser.add_argument('bag_name',  type=str, help="name of the bag file")
parser.add_argument('--split',  action='store_true', help="split the rosbag or not")
args = vars(parser.parse_args())

if __name__ == "__main__":
    print('Parse bag file:', args['bag_name'])
    bag = Bag(args['bag_name'])
    if args['split'] is True:
        print('Convert image message to rgb from', args['bag_name'], 'and split')
        parseRosbag(bag)
    else:
        print('Convert image message to rgb from', args['bag_name'])
        toRGB(bag)


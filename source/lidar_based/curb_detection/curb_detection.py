import os
import time
from os.path import isfile, join
import sys
import threading
import multiprocessing
import ctypes

import glob
import numpy as np
import vg
import open3d
import cv2
import rosbag
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from sensor_msgs.msg import CameraInfo, PointField

from skimage.measure import LineModelND, ransac

parser_dir = os.path.abspath(os.path.join(os.path.dirname(__file__),'..')) + '/ros_velodyne/'
sys.path.insert(0, parser_dir)
from parser import RosbagParser

import argparse

debug_print = False

def data_path_loader(path='/home/rtml/LiDAR_camera_calibration_work/data/data_bag/20190424_pointgrey/'):
    path_horizontal = path + 'horizontal/*.bag'
    path_tilted = path + 'tilted/*.bag'

    horizontal = sorted(glob.glob(path_horizontal))
    tilted = sorted(glob.glob(path_tilted))
    return {'horizontal': horizontal, 'tilted': tilted}

def get_points_from_laser_number(pointcloud, num):
    """
    Get lidar points from laser scan "num"
    
    @param pointcloud: input pointcloud
    @type: numpy array with shape (n, 5)
    @num: number in range(0, 16)
    @type: float
    @return: output pointcloud
    @rtype: numpy array with shape (n, 5)
    """
    idx = pointcloud[:,4] == num
    return pointcloud[idx,:]    

def rearrange_pointcloud_by_ring(pointcloud):
    """
    Rearrange the pointcloud by the order "laser scan 0 to 15"
    
    @param pointcloud: input pointcloud
    @type: numpy array with shape (n, 5)
    @return: output pointcloud
    @rtype: numpy array with shape (n, 5)
    """
    pc_rearrange = np.empty((0, 5), dtype=float)
    index = np.empty((16, 2), dtype=int)
    curr = 0
    for i in range(0, 16):
        pc_i = get_points_from_laser_number(pointcloud, float(i))
        pc_rearrange = np.vstack((pc_rearrange, pc_i))
        index[i] = [curr, pc_rearrange.shape[0]]
        curr = pc_rearrange.shape[0]
    if debug_print:
        for i in range(0, 16):
            print(index[i])
    return pc_rearrange, index

def get_pointcloud_list_by_ring_from_pointcloud(pointcloud, n_result=5):
    """
    Return a pointcloud list by the order "laser scan 0 to 15"
    
    @param pointcloud: input pointcloud
    @type: numpy array with shape (n, 5)
    @return: output list of pointcloud 
    @rtype: list of numpy array 
    """
    pc_list = []
    for i in range(0, 16):
        pc_i = get_points_from_laser_number(pointcloud, float(i))
        curb = np.zeros((pc_i.shape[0],n_result), 'float')
        pc_i = np.hstack((pc_i, curb))
        left_idx = pc_i[:,1] > 0.
        right_idx = pc_i[:,1] <= 0.
        left = pc_i[left_idx]
        right = pc_i[right_idx]
        pc_list.append({'left': left, 'right': right})
    return pc_list

def get_rearranged_pointcloud(pointcloud):
    pointcloud_re = np.empty_like(pointcloud)
    range_idx = []
    cur = 0
    for i in range(0, 16):
        pc_i = get_points_from_laser_number(pointcloud, float(i))
        left = pc_i[pc_i[:,1] > 0.]
        right = pc_i[pc_i[:,1] <= 0.]
        left = reorder_pointcloud(left)
        right = reorder_pointcloud(right)
        len_l, len_r = left.shape[0], right.shape[0]
        pointcloud_re[cur:cur+len_l] = left
        pointcloud_re[cur+len_l:cur+len_l+len_r] = right
        range_idx.append([cur, cur+len_l])
        range_idx.append([cur+len_l, cur+len_l+len_r])
        cur += (len_l + len_r)
    return pointcloud_re, range_idx

"""
color map for laser scan 0 to 15
"""
c_map = np.zeros((16, 3), dtype='float')
c_map[0] = np.array([1., 0., 0.])
c_map[1] = np.array([0., 1., 0.])
c_map[2] = np.array([0., 0., 1.])
c_map[3] = np.array([0.5, 0.5, 0.])
c_map[4] = np.array([0.5, 0., 0.5])
c_map[5] = np.array([0., 0.5, 0.5])
c_map[6] = np.array([1.0, 0.5, 0.])
c_map[7] = np.array([1.0, 0., 0.5])
c_map[8] = np.array([0, 1., 0.5])
c_map[9] = np.array([0.5, 1., 0.])
c_map[10] = np.array([0., 0.5, 1.])
c_map[11] = np.array([0.5, 0.5, 1.])
c_map[12] = np.array([0.5, 1., 0.5])
c_map[13] = np.array([1., 0.5, 0.5])
c_map[14] = np.array([0., 0., 1.])
c_map[15] = np.array([0.5, 0.5, 0.])

def get_color(pointcloud):
    """ 
    Get the color for each point from hardcoded color map for each scan line [0, 15]

    @param pointcloud: input pointcloud
    @type: numpy array with shape (n, 5)
    @return: output color map for the pointcloud
    @rtype: numpy array with shape (n, 3)
    """ 
    n, c = pointcloud.shape
    color = np.zeros((n, 3), dtype='float')
    for i in range(0, 16):
        idx = pointcloud[:,4] == float(i)
        color[idx] = c_map[i] 
    return color

def get_color_from_curb(pointcloud):
    """ 
    Get the color for each point from hardcoded color map for each scan line [0, 15]

    @param pointcloud: input pointcloud
    @type: numpy array with shape (n, 5)
    @return: output color map for the pointcloud
    @rtype: numpy array with shape (n, 3)
    """ 
    n, c = pointcloud.shape
    color = np.zeros((n, 3), dtype='float')
    color[pointcloud[:,9] == 0.] = [0., 0., 0.] 
    color[pointcloud[:,9] == 0.5] = c_map[0] 
    color[pointcloud[:,9] == 1.] = c_map[13] 
    return color

def get_color_elevation(elevation, value=.005):
    """ 
    Get the color for each point by elevation value

    @param elevation: input elevation value for each point
    @type: numpy array with shape (n, 1)
    @param value: threshold
    @type: float
    @return: output color map for the pointcloud
    @rtype: numpy array with shape (n, 3)
    """ 
    n, c = elevation.shape
    color = np.ones((n, 3), dtype='float') / 2.
    idx = (elevation[:, 0] > value) + (elevation[:, 0] < -value) 
    color[idx] = np.array([1., 0., 0.])
    print("Points marked:", np.sum(idx))
    return color

def max_height_filter(pointcloud, max_height):
    """ 
    Filter the pointcloud by maximun height

    @param pointcloud: input pointcloud
    @type: numpy array with shape (n, 5)
    @param max_height: threshold for maximum height
    @type: float
    @return: output filtered pointcloud
    @rtype: numpy array with shape (n, 5)
    """ 
    idx = pointcloud[:,2] < max_height
    return pointcloud[idx,:]    

def FOV_positive_x_filter(pointcloud):
    """ 
    Filter the pointcloud by x value, return only positive x

    @param pointcloud: input pointcloud
    @type: numpy array with shape (n, 5)
    @param max_height: threshold for maximum height
    @type: float
    @return: output filtered pointcloud
    @rtype: numpy array with shape (n, 5)
    """ 
    idx = pointcloud[:,0] > 0.
    return pointcloud[idx,:]    

def get_slope(p1, p2):
    """ 
    Calculate slope of two points p1 and p2

    @param p1: input point
    @type: numpy array with shape (1, 3)
    @param p1: input point
    @type: numpy array with shape (1, 3)
    @return: output slope of vector p1p2
    @rtype: float
    """ 
    dist = np.linalg.norm(p2[0:2]-p1[0:2])
    if debug_print:
        print(p1, p2, dist)
    return (p2[2] - p1[2]) / dist

def get_z_diff(p1, p2):
    """ 
    Calculate z direction difference of two points p1 and p2

    @param p1: input point
    @type: numpy array with shape (1, 3)
    @param p1: input point
    @type: numpy array with shape (1, 3)
    @return: output z value of vector p1p2
    @rtype: float
    """ 
    if debug_print:
        print(p1, p2, dist)
    return p2[2] - p1[2]

def elevation_map(pointcloud):
    """ 
    Return a (n, 1) evevation map for each point in the pointcloud

    @param pointcloud: input pointcloud
    @type: numpy array with shape (n, 5)
    @return: output elevation map
    @rtype: numpy array with shape (n, 1)
    """ 
    n, c = pointcloud.shape
    elevation = np.zeros((n, 1), dtype='float')
    curr_layer = pointcloud[0,4]
    first = 0
    for i in range(0, n):
        if i == n-1:
            elevation[i] = get_slope(pointcloud[i,:3], pointcloud[first, :3])
        elif pointcloud[i+1,4] > curr_layer:
            elevation[i] = get_slope(pointcloud[i,:3], pointcloud[first, :3])
            curr_layer = pointcloud[i+1,4]
            first = i+1
        else:
            elevation[i] = get_slope(pointcloud[i,:3], pointcloud[i+1, :3])
    return elevation            

def add_curb_column(pointcloud, elevation, value=.005):
    """ 
    Get the color for each point by elevation value

    @param pointcloud: input pointcloud
    @type: numpy array with shape (n, 5)
    @param elevation: input elevation values
    @type: numpy array with shape (n, 1)
    @param value: threshold
    @type: float
    @return: output pointcloud with one more column "curb"
    @rtype: numpy array with shape (n, 6)
    """ 
    n, c = elevation.shape
    curb = np.zeros_like(elevation)
    idx = (elevation[:, 0] > value) + (elevation[:, 0] < -value) 
    curb[idx] = np.array([1.])
    return np.hstack((pointcloud, curb)) 

def test_visualize(data):
    """ 
    Visualize the pointcloud data from the RosbagParser object "lidar_data" in open3d visualizer
    @param data: pointcloud data
    @type: np array with shape = (n, 5), for each row: (x, y, z, i, r)
    """
    # data_xyz = data[:,:3]
    # data_ir = data[:,3:]
    # print(data_ir.shape)
    pc_list = []
    for i in range(0, 16):
        pc_i = get_points_from_laser_number(data, float(i))
        pc_list.append(pc_i)
    
    pc_rearrange = rearrange_pointcloud_by_ring(data)
    # pc_rearrange = max_height_filter(pc_rearrange, -0.9)
    
    data_xyz = pc_rearrange[:,:3]
    color_map = get_color(pc_rearrange)

    vis = open3d.Visualizer()
    vis.create_window()
    pcd = open3d.PointCloud()
    pcd.points = open3d.Vector3dVector(data_xyz)
    vis.add_geometry(pcd)
    
    n = data_xyz.shape[0] # n = number of points
    for idx in range(0, n): 
        # Visualizing lidar points in camera coordinates
        pcd.points = open3d.Vector3dVector(data_xyz[0:idx,:])
        pcd.colors = open3d.Vector3dVector(color_map[0:idx,:])
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
    vis.destroy_window()

def update_vis(vis, pcd, pointcloud, color_map):
    """
    Update the open3d visualizer in the loop
    """
    pcd.points = open3d.Vector3dVector(pointcloud)
    pcd.colors = open3d.Vector3dVector(color_map)
    vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()

def add_origin_axis(vis, z=0.9):
    """
    Draw the xyz coordinates axis in open3d visualizer at the origin / (0, 0, z)
    """
    points = [[0,0,0-z],[1,0,0-z],[0,1,0-z],[0,0,1-z]]
    lines = [[0,1],[0,2],[0,3]]
    colors = [[1,0,0],[0,1,0],[0,0,1]]
    line_set = open3d.LineSet()
    line_set.points = open3d.Vector3dVector(points)
    line_set.lines = open3d.Vector2iVector(lines)
    line_set.colors = open3d.Vector3dVector(colors)
    vis.add_geometry(line_set)

def get_pointcloud_from_msg(msg):
    """
    Get pointcloud from pointcloud2 ros message

    @param msg: ros message
    @type: pointcloud2
    @return: output pointcloud 
    @rtype: numpy array with shape (n, 5)
    """
    pc_list = list(pc2.read_points(msg))
    return np.array(pc_list, 'float32')

def get_pointcloud_list_from_msg(msg):
    """
    Get pointcloud list from pointcloud2 ros message

    @param msg: ros message
    @type: pointcloud2
    @return: output pointcloud list 
    @rtype: list of numpy array 
    """
    pc_list = list(pc2.read_points(msg))
    pointcloud = np.array(pc_list, 'float32')
    pointcloud_list = []
    for i in range(0, 16):
        pointcloud_list.append(get_points_from_laser_number(pointcloud, float(i)))
    return pointcloud_list

def rotation_matrix(theta=90.):
    """
    Return a rotation matrix which rotata CCW along y axis

    @param theta: theta value in degree
    @type: float
    @return: output rotation matrix
    @rtype: numpy array with shape (3, 3)
    """
    theta = theta * np.pi / 180.
    return np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]], 'float32')


def rotate_pc(pointcloud, rot):
    """
    Return pointcloud rotated with rotation matrix rot

    @param pointcloud: input pointcloud
    @type: numpy array with shape (n, 5)
    @param rot: input rotation matrix
    @type: numpy array with shape (3, 3)        get_points_from_laser_number()
    @return: output rotated pointcloud
    @rtype: numpy array with shape (n, 5        get_points_from_laser_number()
    """
    pc_trans = np.transpose(pointcloud[:,:3])
    pc_rotated = np.matmul(rot, pc_trans)
    pc_rotated = np.transpose(pc_rotated)
    pointcloud[:,:3] = pc_rotated
    return pointcloud

def translate_z_pc(pointcloud, z):
    """
    Return pointcloud traslated with z

    @param pointcloud: input pointcloud
    @type: numpy array with shape (n, 5)
    @param z: input traslation value in z direction
    @type: float
    @return: output translated pointcloud
    @rtype: numpy array with shape (n, 5)
    """
    pointcloud[:,2] += z
    return pointcloud

def find_matrix(lidar_data):
    z = 0.
    rot = rotation_matrix()
    for topic_1, msg_1, t_1 in lidar_data.topic_1:
        pointcloud = get_pointcloud_from_msg(msg_1)
        pc_i = get_points_from_laser_number(pointcloud, 0)
        for i in range(0, 20):
            theta = 15. + 0.5 * i
            print(theta)
            rot = rotation_matrix(theta)
            pc_new = rotate_pc(pc_i, rot) 
            print(pc_new[:,2])
            plt.hist(pc_new[:,2], bins=50)    
            plt.show() 
        break

    return rot, z

def visualize_from_bag(lidar_data, config='horizontal'):
    """
    Visualize the pointcloud data from the RosbagParser object "lidar_data" in open3d visualizer

    @param lidar_data: input lidar data  
    @type: RosbagParser object
    @param config: input config of the lidar sensor  
    @type: string
    @return: 
    @rtype: 
    """
    if config not in ['horizontal', 'tilted']:
        print('Invalid config input, should be horizontal or tilted')
        return

    # initialize visualizer
    vis = open3d.Visualizer()
    vis.create_window(window_name='point cloud', width=1280, height=960)
    pcd = open3d.PointCloud()
    ctr = vis.get_view_control()

    # draw coodinate axis at (0, 0, -0.9)
    add_origin_axis(vis)

    # get rotation matrix
    rot = rotation_matrix(18.)

    idx = 0
    for topic_1, msg_1, t_1 in lidar_data.topic_1:
        print('frame', idx, '/', lidar_data.len_1)
        # get pointcloud from current frame
        pointcloud = get_pointcloud_from_msg(msg_1)
        # pc_rearrange = get_points_from_laser_number(pointcloud, 0)
        pc_rearrange = rearrange_pointcloud_by_ring(pointcloud)
        if config == 'tilted': 
            pc_rearrange = rotate_pc(pc_rearrange, rot)
        pc_rearrange = translate_z_pc(pc_rearrange, 0.3)
        pc_rearrange = max_height_filter(pc_rearrange, .3)

        # calculate elevation    
        elevation =  elevation_map(pc_rearrange)

        if config == 'tilted': 
            color_map = get_color_elevation(elevation, 0.01)
        else:
            color_map = get_color_elevation(elevation, 0.003)

        # visualizing lidar points in camera coordinates
        if idx == 0:
            pcd.points = open3d.Vector3dVector(pc_rearrange[:,:3])
            vis.add_geometry(pcd)
        update_vis(vis, pcd, pc_rearrange[:,:3], color_map)
        idx += 1
    vis.destroy_window()

def pc2_message(msg, pc_data):
    header = Header()
    header.frame_id = msg.header.frame_id
    header.stamp = msg.header.stamp

    # add one more field 'curb' in the pointcloud2 message
    fields = [PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('intensity', 12, PointField.FLOAT32, 1),
            PointField('ring', 16, PointField.FLOAT32, 1),
            PointField('curb', 20, PointField.FLOAT32, 1)]
    col = pc_data.shape[1]
    if (col > 6):
        for i in range(1, col-5):
            fields.append(PointField('curb'+str(i), 20+4*i, PointField.FLOAT32, 1))
    return pc2.create_cloud(header, fields, pc_data)

def edge_filter_v1(pointcloud_z, index, k=4):
    """
    Calculate z difference from current point to k points left and k points right
    If one side close to zero and the other is not, mark it as edge point

    @param pointcloud: input pointcloud with only z 
    @type: numpy array with shape (n, 1)
    @param index: index recording start and end position of each ring from 0 to 15
    @type: numpy array with shape (16, 2)
    @return:  
    @rtype: 
    """
    thres = 0.05
    edges = np.zeros((pointcloud_z.shape[0], 1), 'float')
    for r in range(0,16):
        start, end = index[r,0], index[r,1]
        pc_i = pointcloud_z[start:end]   #               shape = (n, 1)
        diff1 = pc_i[1:]-pc_i[:-1]        # (i-1) - i    shape = (n-1, 1)
        diff2 = pc_i[2:]-pc_i[:-2]        # (i-2) - i    shape = (n-2, 1)
        diff3 = pc_i[3:]-pc_i[:-3]        # (i-3) - i    shape = (n-3, 1)
        diff4 = pc_i[4:]-pc_i[:-4]        # (i-4) - i    shape = (n-4, 1)
        # diff5 = pc_i[5:]-pc_i[:-5]        # (i-5) - i    shape = (n-5, 1)
        right_sum = diff1[k:-3] + diff2[k:-2] + diff3[k:-1] + diff4[k:]
        left_sum = (-diff1[3:-k]) + (-diff2[2:-k]) + (-diff3[1:-k]) + (-diff4[:-k])
        right_sum = np.abs(right_sum) 
        left_sum = np.abs(left_sum)
        is_edge = (right_sum < thres) * (left_sum > thres) + (right_sum > thres) * (left_sum < thres)
        edges[start+k: end-k][is_edge] = 1.
    return edges

def curb_detection_v1(msg, config, rot, height):
    """
    Detect and return ros message with additional "curb" information  
    Version one

    @param msg: input ros message
    @type: pointcloud2
    @param config: input config of the lidar sensor  
    @type: string
    @param rot: input rotation matrix
    @type: numpy array with shape (3, 3)
    @param height: input translation value in z direction
    @type: float
    @return: output ros message 
    @rtype: ros pointcloud2 message
    """
    pointcloud = get_pointcloud_from_msg(msg)

    if config == 'tilted': 
        pointcloud = rotate_pc(pointcloud, rot)
        pointcloud = translate_z_pc(pointcloud, height)
    else:
        pointcloud = translate_z_pc(pointcloud, height-0.2)
    pointcloud = max_height_filter(pointcloud, .3)
    pointcloud = FOV_positive_x_filter(pointcloud)
    pc_rearrange, index = rearrange_pointcloud_by_ring(pointcloud)

    # calculate elevation    
    elevation =  elevation_map(pc_rearrange)

    # calculate z difference for 5 points left and 5 points right at point i
    edges = edge_filter_v1(pc_rearrange[:,2], index)
    pc_data = add_curb_column(pc_rearrange, edges, 0.5)

    # if config == 'tilted': 
    #     pc_data = add_curb_column(pc_rearrange, elevation, 0.5)
    # else:
    #     pc_data = add_curb_column(pc_rearrange, elevation, 0.003)
    return pc2_message(msg, pc_data)

# looks fine with thres_slope=0.07 thres_flat=0.04 (fails when curb is not flat)
def find_edge_from_right_half_v02(pointcloud_right, k=6, thres_slope=0.07,thres_flat=0.04):
    """
    Detect and return edge points index from single laser group at right side of vehicle

    @param pointcloud: input pointcloud (right & front in CW order) 
    @type: numpy array with shape (n, 6)
    @return: index of curb points 
    @rtype: numpy array with shape (n, 6)
    """
    n =  pointcloud_right.shape[0]
    if n - 2 * k < 0:
        return pointcloud_right

    # reorder the points
    theta = np.zeros(n, 'float') 
    for i in range(n):
        theta[i] = -np.arctan2(pointcloud_right[i,1], pointcloud_right[i,0]) * 180 / np.pi 
    order = np.argsort(theta)
    pointcloud_right = pointcloud_right[order,:]

    # calculate left_sum and right_sum
    right_sum, left_sum = np.zeros((n-2*k), 'float'), np.zeros((n-2*k), 'float')
    diff = []
    pointcloud_z = pointcloud_right[:,2]
    for i in range(1,k+1):
        diff1 = pointcloud_z[i:]-pointcloud_z[:-i] 
        diff.append(diff1)
    for i in range(k):
        if i == k - 1:
            right_sum += diff[i][k:]
            left_sum += (-diff[i][:-k])
        else:
            right_sum += diff[i][k:(i-k+1)] 
            left_sum += (-diff[i][k-i-1:-k])

    # parameters
    curr_start, curr_end = 0, 0

    # find start points    
    is_edge_start = (right_sum > thres_slope) * (np.abs(left_sum) < thres_flat)
    is_edge_start = np.pad(is_edge_start, (k,k), 'constant',constant_values=False)
    pointcloud_right[:,5][is_edge_start] = 1.
    
    # find end points    
    is_edge_end = (left_sum < -thres_slope) * (np.abs(right_sum) < thres_flat)
    is_edge_end = np.pad(is_edge_end, (k,k), 'constant',constant_values=False)
    pointcloud_right[:,5][is_edge_end] = 0.4

    curb_list = []
    for i in range(n):
        if is_edge_start[i]:
            curr_start = i
        if is_edge_end[i] and curr_start != 0:
            h =  pointcloud_z[i] - pointcloud_z[curr_start]
            if h > 0.02 and h < 0.3:
                curb_list.append([curr_start, i])
                curr_start, curr_end = 0, 0
                curb_height = pointcloud_z[i]

    for c in curb_list:
        if c[0] < c[1] and c[0] > 0 and c[1] > 0 and c[1]-c[0] > 4:
            pointcloud_right[c[0]+1:c[1],5] = 0.7  
    
    # first_start =  np.argmax(pointcloud_right[:,5] > 0.5) # !!! would return 0 if nothing found
    # first_end =  np.argmax((pointcloud_right[:,5] > 0.3) * (pointcloud_right[:,5] < 0.5)) # !!! would return 0 if nothing found
    # if first_start < first_end:
    #     pointcloud_right[first_start+1:first_end,5] = 0.7  
    return pointcloud_right

def find_edge_from_left_half_v02(pointcloud_left, k=4, thres_slope=0.05,thres_flat=0.02):
    """
    Detect and return edge points index from single laser group at left side of vehicle

    @param pointcloud: input pointcloud (left & front in CW order) 
    @type: numpy array with shape (n, 6)
    @return: index of curb points 
    @rtype: numpy array with shape (n, 6)
    """
    n =  pointcloud_left.shape[0]
    if n - 2 * k < 0:
        return pointcloud_left
    
    # calculate left_sum and right_sum
    right_sum, left_sum = np.zeros((n-2*k), 'float'), np.zeros((n-2*k), 'float')
    diff = []
    pointcloud_z = pointcloud_left[:,2]
    for i in range(1,k+1):
        diff1 = pointcloud_z[i:]-pointcloud_z[:-i] 
        diff.append(diff1)
    for i in range(k):
        if i == k - 1:
            right_sum += diff[i][k:]
            left_sum += (-diff[i][:-k])
        else:
            right_sum += diff[i][k:(i-k+1)] 
            left_sum += (-diff[i][k-i-1:-k])
    
    # find start points    
    is_edge_start = (left_sum > thres_slope) * (np.abs(right_sum) < thres_flat)
    is_edge_start = np.pad(is_edge_start, (k,k), 'constant',constant_values=False)
    
    pointcloud_left[:,5][is_edge_start] = 1.
    
    # find end points    
    is_edge_end = (right_sum < -thres_slope) * (np.abs(left_sum) < thres_flat)
    is_edge_end = np.pad(is_edge_end, (k,k), 'constant',constant_values=False)
    
    pointcloud_left[:,5][is_edge_end] = 0.4

    # parameters
    max_curb = 0.3
    curb_height = 0
    curr_start, curr_end = 0, 0
    curb_list = []
    for i in range(n-1,-1,-1):
        if is_edge_start[i]:
            if curr_start == 0:
                curr_start = i
            else:
                if curr_end == 0:
                    curr_start = i
                else:
                    pass
        
        if is_edge_end[i] and curr_start != 0:
            h =  pointcloud_z[i] - pointcloud_z[curr_start]
            if h > 0.05 and h < 0.3:
                curb_list.append([curr_start, i])
            curr_start, curr_end = 0, 0
            curb_height = pointcloud_z[i]

    for c in curb_list:
        if c[1] < c[0] and c[0] > 0 and c[1] > 0 and c[0]-c[1] > 6:
            pointcloud_left[c[1]+1:c[0],5] = 0.7    

    return pointcloud_left

def reorder_pointcloud(pointcloud):
    n = pointcloud.shape[0]
    theta = np.zeros(n, 'float') 
    # for i in range(n):
    #     theta[i] = -np.arctan2(pointcloud[i,1], pointcloud[i,0]) * 180 / np.pi 
    theta = -np.arctan2(pointcloud[:,1], pointcloud[:,0]) * 180 / np.pi 
    order = np.argsort(theta)
    return pointcloud[order,:]

def unit_vec(vec):
    return vec / np.linalg.norm(vec)

def angle_between(v1, v2):
    v1_unit = unit_vec(v1) 
    v2_unit = unit_vec(v2) 
    return np.arccos(np.clip(np.dot(v1_unit, v2_unit), -1., 1.))

def edge_filter(pointcloud, k, half, thres_slope, thres_flat, thres_h_low=0.09, thres_h_high=0.25):
    """
    Calculate z difference from current point to k points left and k points right
    If one side close to zero and the other is not, mark it as edge point

    @param pointcloud: input pointcloud with only z 
    @type: numpy array with shape (n, 1)
    @param index: index recording start and end position of each ring from 0 to 15
    @type: numpy array with shape (16, 2)
    @return:  
    @rtype: 
    """
    n =  pointcloud.shape[0]
    res = np.zeros((n,1),'float')
    
    # calculate left_sum and right_sum
    right_sum, left_sum = np.zeros((n-2*k), 'float'), np.zeros((n-2*k), 'float')
    diff = []
    pointcloud_z = pointcloud[:,2]
    for i in range(1,k+1):
        diff1 = pointcloud_z[i:]-pointcloud_z[:-i] 
        diff.append(diff1)
    for i in range(k):
        if i == k - 1:
            right_sum += diff[i][k:]
            left_sum += (-diff[i][:-k])
        else:
            right_sum += diff[i][k:(i-k+1)] 
            left_sum += (-diff[i][k-i-1:-k])

    # parameters
    curr_start, curr_end = 0, 0

    if half == 'left':
        idx = range(n-1,-1,-1)
        # find start points    
        is_edge_start = (left_sum > thres_slope) * (np.abs(right_sum) < thres_flat)
        is_edge_start = np.pad(is_edge_start, (k,k), 'constant',constant_values=False)
        
        # find end points    
        is_edge_end = (right_sum < -thres_slope) * (np.abs(left_sum) < thres_flat)
        is_edge_end = np.pad(is_edge_end, (k,k), 'constant',constant_values=False)
    else:
        idx = range(n)
        # find start points    
        is_edge_start = (right_sum > thres_slope) * (np.abs(left_sum) < thres_flat)
        is_edge_start = np.pad(is_edge_start, (k,k), 'constant',constant_values=False)
        
        # find end points    
        is_edge_end = (left_sum < -thres_slope) * (np.abs(right_sum) < thres_flat)
        is_edge_end = np.pad(is_edge_end, (k,k), 'constant',constant_values=False)

    res[is_edge_start] = 1.
    res[is_edge_end] = 0.4

    curb_list = []
    for i in idx:
        if is_edge_start[i]:
            curr_start = i
        if is_edge_end[i] and curr_start != 0:
            h =  pointcloud_z[i] - pointcloud_z[curr_start]
            if h > thres_h_low and h < thres_h_high and abs(i - curr_start) > 4:
                if i > curr_start:
                    curb_list.append([curr_start, i])
                else:
                    curb_list.append([i, curr_start])
                curr_start, curr_end = 0, 0
                curb_height = pointcloud_z[i]
    return curb_list

def edge_filter_v2(pointcloud, ground, k, half, thres_slope, thres_flat, thres_h_low=0.09, thres_h_high=0.25):
    """
    Calculate z difference from current point to k points left and k points right
    If one side close to zero and the other is not, mark it as edge point

    @param pointcloud: input pointcloud with only z 
    @type: numpy array with shape (n, 1)
    @param index: index recording start and end position of each ring from 0 to 15
    @type: numpy array with shape (16, 2)
    @return:  
    @rtype: 
    """
    n =  pointcloud.shape[0]
    res = np.zeros((n,1),'float')
    
    # calculate left_sum and right_sum
    right_sum, left_sum = np.zeros((n-2*k), 'float'), np.zeros((n-2*k), 'float')
    diff = []
    pointcloud_z = pointcloud[:,2]
    for i in range(1,k+1):
        diff1 = pointcloud_z[i:]-pointcloud_z[:-i] 
        diff.append(diff1)
    for i in range(k):
        if i == k - 1:
            right_sum += diff[i][k:]
            left_sum += (-diff[i][:-k])
        else:
            right_sum += diff[i][k:(i-k+1)] 
            left_sum += (-diff[i][k-i-1:-k])

def line_fitting_filter(pointcloud_right, curb_list):
    dis_thres = 0.03
    curb2 = np.zeros((pointcloud_right.shape[0],1),'float')
    for c in curb_list:
        curb2[c[0]] = 1.
        curb2[c[1]] = 0.4
        x_i = pointcloud_right[c[0]+1:c[1],0]
        y_i = pointcloud_right[c[0]+1:c[1],1]
        x = pointcloud_right[[c[0],c[1]],0]
        y = pointcloud_right[[c[0],c[1]],1]
        param = np.polyfit(x, y, 1)
        n = np.sqrt(param[0]*param[0] + 1.)
        isline = True
        dis = np.abs(x_i * param[0] + param[1] - y_i) / n
        idx = dis > dis_thres
        if (sum(idx) < 5):
            curb2[c[0]+1:c[1]] = 0.7  
        # for i in range(x_i.shape[0]):
        #     # print(np.abs(x[i] * param[0] + param[1] - y[i]) / n)
        #     if np.abs(x_i[i] * param[0] + param[1] - y_i[i]) / n > dis_thres:
        #         isline = False
        #         break
        # if isline:
        #     curb2[c[0]+1:c[1]] = 0.7  
    return np.hstack((pointcloud_right, curb2))

def direction_change_filter(pointcloud, k=8, angle_thres=150.):
    """
    Detect and return the angle between left and right side of a point

    @param pointcloud: input pointcloud (left or right & front in CW order) 
    @type: numpy array with shape (n, x)
    @return: angle of each point in degree 
    @rtype: 1d numpy array with shape (n, )
    """
    n = pointcloud.shape[0]
    result = np.zeros((n,1),'float')
    if n - 2 * k < 0:
        return result
    pointcloud_xyz = pointcloud[:,:3]
    direction = pointcloud_xyz[1:]-pointcloud_xyz[:-1] 

    right_sum, left_sum = np.zeros((n-2*k,3), 'float'), np.zeros((n-2*k,3), 'float')
    diff = []
    for i in range(1,k+1):
        diff1 = pointcloud_xyz[i:]-pointcloud_xyz[:-i]
        length = np.linalg.norm(diff1, axis=1)
        diff.append(diff1)
        # diff.append(diff1/length.reshape(-1,1)) # normalize the vectors
    for i in range(k):
        # add weights on directions
        if i == k - 1:
            right_sum += diff[i][k:] * (i+1)
            left_sum += (-diff[i][:-k]) * (i+1)
        else:
            right_sum += diff[i][k:(i-k+1)] * (i+1)
            left_sum += (-diff[i][k-i-1:-k]) * (i+1)

    angles = vg.angle(right_sum, left_sum)
    angles = np.pad(angles, (k,k), 'constant',constant_values=180.)
    return angles

def local_min_of_direction_change(pointcloud, half):
    """
    Find the local min points from direction change filter

    @param pointcloud: input pointcloud (left or right & front in CW order) 
    @type: numpy array with shape (n, 6)
    @param half: input string indicating left or right pointcloud 
    @type: string
    @return: index of curb points 
    @rtype: numpy array with shape (n, 6) or (n, 7) if comparision needed
    """
    direction = direction_change_filter(pointcloud)
    filter_conv = [1., 1., 1., 1., 1.]
    thres = 3.
    direction_change = np.where(direction < 150., 1., 0.)
    direction_change = np.convolve(direction_change.ravel(), filter_conv, 'same')
    direction_change = np.where(direction_change >= thres, 1., 0.)

    # local min of direction change
    local_min = np.concatenate(([0.], np.diff(np.sign(np.diff(direction))), [0.]))
    local_min_index = (local_min > 0.) * (direction < 150.) 
    return local_min_index

def edge_filter_from_elevation(pointcloud, elevation, half):
    """
    Find the possible edge start and edge end points from elevation

    @param pointcloud: input pointcloud (left or right & front in CW order) 
    @type: numpy array with shape (n, x)
    @param elevation: input elevation information 
    @type: 1d numpy array with shape (n, )
    @param half: input string indicating left or right pointcloud 
    @type: string
    @return: index of possible edge start and edge end 
    @rtype: two numpy array with shape (n, x) or (n, x)
    """
    if half == 'left':
        f_start = [-1,-1,-1,-1,-1,-1,-1,1,1,1,1,1,1,1,1] 
        f_end = [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0] 
    else:
        f_start = [1,1,1,1,1,1,1,1,-1,-1,-1,-1,-1,-1,-1] 
        f_end = [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1] 
    edge_start = np.convolve(elevation, f_start, 'same') >= 2.
    edge_end = np.convolve(elevation, f_end, 'same') >= 6.
    return edge_start, edge_end

sensor_height = 1.195 
tilted_angle = 19.2
theta_r = 0.00356999
angles = [-15., -13., -11., -9., -7., -5., -3., -1., 1., 3., 5., 7., 9., 11., 13., 15.]
angles_tilted = [-(i - tilted_angle) * np.pi / 180. for i in angles]
# dis_thres = [sensor_height / np.sin(i) * theta_r * 1.3 for i in angles_tilted]
dis_thres = [theta_r * 1.3 for i in angles_tilted]
def elevation_filter(pointcloud, half):
    """
    Detect and return index of points which the distance to left and right neighbors
    is higher than the threshold

    @param pointcloud: input pointcloud (left or right & front in CW order) 
    @type: numpy array with shape (n, 6)
    @return: index of curb points 
    @rtype: numpy array with shape (n, 6) or (n, 7) if comparision needed
    """
    ground = np.zeros(pointcloud.shape[0],'float')
    pointcloud_xyz = pointcloud[:,:3]
    dist_to_origin = np.linalg.norm(pointcloud_xyz, axis=1)
    thres = dist_to_origin * theta_r

    diff = pointcloud_xyz[1:]-pointcloud_xyz[:-1] 
    diff_z = diff[:,2]
    diff_r = diff_z[1:]
    diff_l = diff_z[:-1]
    diff_r = np.pad(diff_r, (1,1), 'constant',constant_values=0.)
    diff_l = np.pad(diff_l, (1,1), 'constant',constant_values=0.)
    # dist_r = np.linalg.norm(diff_r, axis=1)
    # dist_l = np.linalg.norm(diff_l, axis=1)
    # dist = dist_r + dist_l
    # dist = np.pad(dist, (1,1), 'constant',constant_values=0.)

    dist_z = diff_r+diff_l
    if half == 'left': 
        dist_z = -dist_z

    # ground[dist_z > 0.01] = 1.
    filter_conv = [1., 1., 1., 1., 1., 1., 1., 1., 1.]
    thres = 4.
    dist_z =  np.convolve(np.where(dist_z > 0.008, 1., 0.), filter_conv, 'same')
    dist_z =  np.convolve(np.where(dist_z >= thres, 1., 0.), filter_conv, 'same')
    # if half == 'right':
    #     dist_z =  np.convolve(np.where(dist_z > 0.01, 1., 0.), [1., 1., 1., 1., 1., 0., 0., 0., 0.], 'same')
    # else:
    #     dist_z = np.convolve(np.where(dist_z > 0.01, 1., 0.), [0., 0., 0., 0., 1., 1., 1., 1., 1.], 'same')
    ground[dist_z >= thres] = 1.
    return ground

def continuous_filter(pointcloud):
    """
    Return distance between point i and i+1 

    @param pointcloud: input pointcloud (left or right & front in CW order) 
    @type: numpy array with shape (n, 6)
    @return: distance between point i and i+1 
    @rtype: 1d numpy array 
    """
    pointcloud_xyz = pointcloud[:,:3]
    dist_to_origin = np.linalg.norm(pointcloud_xyz, axis=1)
    thres = dist_to_origin * theta_r * 7

    diff_r = np.linalg.norm(pointcloud_xyz[1:]-pointcloud_xyz[:-1], axis=1)
    diff_l = np.linalg.norm(pointcloud_xyz[:-1]-pointcloud_xyz[1:], axis=1)
    diff_r = np.pad(diff_r, (0,1), 'constant',constant_values=0.)
    diff_l = np.pad(diff_l, (1,0), 'constant',constant_values=0.)
    return (diff_r > thres) + (diff_l > thres)

def xxx_filter(pointcloud, curb_list):
    def dis_in_thres(curb, thres):
        dis = np.linalg.norm(pointcloud[curb[0],0:3]-pointcloud[curb[1],0:3])
        return dis >= thres[0] and dis <= thres[1] 
    ring = pointcloud[0,4]
    angle = angles_tilted[int(ring)]
    thres = np.array([min_curb_height, max_curb_height]) / np.sin(angle)
    for curb in curb_list:
        dis = np.linalg.norm(pointcloud[curb[0],0:3]-pointcloud[curb[1],0:3])
    curb_list = [curb for curb in curb_list if dis_in_thres(curb, thres)]  
    return curb_list

angles_tilted = [-(i - tilted_angle) * np.pi / 180. for i in angles]
theoretical_dist = [sensor_height / np.tan(i) for i in angles_tilted]
'''
[1.758389102404863, 1.8976277964159545, 2.0532155737630906, 2.2286655815899494, 
 2.4285606747473447, 2.6589955890283625, 2.9282581114767763, 3.2479148905217103, 
 3.6346131485375346, 4.113216979119717, 4.722594018009133, 5.527093901251933, 
 6.641530768832606, 8.292725502505117, 11.000161937824087, 16.272803632868]
'''
def ground_extraction(pointcloud):
    # height threshold
    H_th, H_max = 0.1, 0.4
    # projected into grid
    resolution, r, c = 0.5, 40, 80
    grid = np.zeros((r, c), 'int')
    min_scan = np.full((r, c), -1., 'float')
    max_h = np.full((r, c), -1., 'float')

    for i in range(pointcloud.shape[0]):
        x_idx, y_idx = int(pointcloud[i,0] / 0.5), int((pointcloud[i,1] + 20) / 0.5)
        if x_idx < 40 and y_idx < 80 and y_idx >= 0: 
            if pointcloud[i,2] > H_max: 
                pointcloud[i,9] = 1.
                continue
            if grid[x_idx][y_idx] == 0: 
                max_h[x_idx][y_idx] = pointcloud[i,2]
                min_scan[x_idx][y_idx] = pointcloud[i,4]
            if pointcloud[i][4] > min_scan[x_idx][y_idx] and pointcloud[i,2] - max_h[x_idx][y_idx] > H_th:
                pointcloud[i,9] = 1.
            else:
                max_h[x_idx][y_idx] = np.max([max_h[x_idx][y_idx], pointcloud[i,2]])
                grid[x_idx][y_idx] += 1
    return pointcloud

theoretical_dist_2 = [sensor_height / np.sin(i) for i in angles_tilted]
def obstacle_extraction(pointcloud_i, continuous, side):
    """
    Return points classified as obstacles 

    @param pointcloud_i: input pointcloud with only scan i (left or right & front in CW order) 
    @type: numpy array with shape (n, x)
    @return: pointcloud_i with a column showing the result 
    @rtype: 2d numpy array 
    """
    n, ring = pointcloud_i.shape[0], int(pointcloud_i[0,4])
    dist = np.linalg.norm(pointcloud_i[:,0:3], axis=1)
    height_thres = 0.3
    cur_start, cur_end = -1, -1
    dis = 0
    if side == "right":
        i = 0
        while i < n:
            if i == 0 and pointcloud_i[i,2] > height_thres:
                cur_start = i
                dis = theoretical_dist_2[ring]
            elif cur_start == -1 and i - 1 >= 0 and continuous[i-1] and continuous[i] and dist[i-1] > dist[i]: 
                cur_start = i
                dis = dist[i-1]
            elif cur_start != -1 and dist[i] > dis: 
                pointcloud_i[cur_start:i,9] = 1.
                cur_start = -1
            if i == n-1 and cur_start != -1:
                pointcloud_i[cur_start:i,9] = 1.
            i += 1
    else:
        i = n - 1
        while i >= 0:
            if i == n - 1 and pointcloud_i[i,2] > height_thres:
                cur_start = i
                dis = theoretical_dist_2[ring]
            elif cur_start == -1 and i + 1 < n and continuous[i+1] and continuous[i] and dist[i+1] > dist[i]: 
                cur_start = i
                dis = dist[i+1]
            elif cur_start != -1 and dist[i] > dis: 
                pointcloud_i[i+1:cur_start+1,9] = 1.
                cur_start = -1
            if i == 0 and cur_start != -1:
                pointcloud_i[i+1:cur_start+1,9] = 1.
            i -= 1
    return pointcloud_i

def find_curb_from_half_v03(pointcloud, half, compare, k=8, thres_slope=0.08, thres_flat=0.06, n_result=4):
    """
    Detect and return curb points index from single laser group at left or right side of vehicle

    @param pointcloud: input pointcloud (left or right & front in CW order) 
    @type: numpy array with shape (n, 6)
    @return: index of curb points 
    @rtype: numpy array with shape (n, 6) or (n, 7) if comparision needed
    """
    # return if not enough points
    n =  pointcloud.shape[0]
    if n - 2 * k < 0:
        if compare:
            return np.hstack((pointcloud, np.zeros((n, n_result-1), 'float')))
        else:
            return pointcloud

    # reorder the points
    pointcloud = reorder_pointcloud(pointcloud)

    # apply edge filter to get possible curb list
    curb_list = edge_filter(pointcloud, k, half, thres_slope, thres_flat)
    for c in curb_list:
        if c[0] < c[1] and c[0] > 0 and c[1] > 0:
            pointcloud[c[0],5] = 1.
            pointcloud[c[1],5] = 0.4
            pointcloud[c[0]+1:c[1],5] = 0.7  
    
    # add line fitting filter
    if compare:
        pointcloud = line_fitting_filter(pointcloud, curb_list)

    ground = elevation_filter(pointcloud, half)
    direct = direction_change_filter(pointcloud)
    # pointcloud = np.hstack((pointcloud, ground))
    pointcloud = np.hstack((pointcloud, ground))
    pointcloud = np.hstack((pointcloud, direct))
    return pointcloud

min_curb_height, max_curb_height = 0.05, 0.2 
def find_curb_from_half_v031(pointcloud, half, k=8, n_result=5):
    """
    Detect and return curb points index from single laser group at left or right side of vehicle
    with elevation filter / direction change filter

    @param pointcloud: input pointcloud (left or right & front in CW order) 
    @type: numpy array with shape (n, 5+n_result)
    @return: index of curb points 
    @rtype: numpy array with shape (n, 5+n_result)
    """
    
    # parameters
    curbs = np.empty((0,5),'float') # np array for detection result of curb points
    c_list = np.empty((0,2),'int')
    curr_start, curr_end, curr_height = 0, 0, 0
    
    # return if not enough points
    n =  pointcloud.shape[0]
    if n - 2 * k < 0:
        return pointcloud, curbs, np.array([0. ,0., 0., 0., 0., 0., 0.])

    t_now = time.time()
    # reorder the points
    pointcloud = reorder_pointcloud(pointcloud)
    t_reorder = time.time() - t_now
    
    t_now = time.time()
    # elevation filter
    elevation = elevation_filter(pointcloud, half)
    t_elevation = time.time() - t_now
    
    t_now = time.time()
    # possible edge start and end from elevation info
    edge_start, edge_end = edge_filter_from_elevation(pointcloud, elevation, half)
    t_edge = time.time() - t_now

    t_now = time.time()
    # local min of direction change
    local_min_index = local_min_of_direction_change(pointcloud, half)
    t_local_min = time.time() - t_now
    
    t_now = time.time()
    # continuous filter
    conti = continuous_filter(pointcloud)
    t_conti = time.time() - t_now

    t_now = time.time()
    # main loop
    if half == 'left':
        i = n-1
        while i >= 0:
            if local_min_index[i] and edge_start[i]:
                curr_start = i
                missed = 0
                while i-1 >= 0:
                    if local_min_index[i] and edge_start[i] and curr_height < min_curb_height:
                        curr_start = i
                    curr_end = i
                    if elevation[i] == 0: 
                        missed += 1
                    missed_rate =  float(missed) / (curr_start-curr_end+1)
                    curr_height = pointcloud[curr_end,2] - pointcloud[curr_start,2]
                    if (missed > 10 and missed_rate > 0.3) or curr_height > max_curb_height or conti[i]:
                        break
                    if edge_end[i] and local_min_index[i]:
                        if curr_height > min_curb_height and curr_height < max_curb_height:
                            c_list = np.vstack((c_list,[[curr_end, curr_start]]))
                            break
                    i -= 1
                curr_start, curr_end, curr_height = 0, 0, 0
            i -= 1
    else:
        i = 0
        while i < n:
            if local_min_index[i] and edge_start[i]:
                curr_start = i
                missed = 0
                while i+1 < n:
                    if local_min_index[i] and edge_start[i] and curr_height < min_curb_height:
                        curr_start = i
                    curr_end = i
                    if elevation[i] == 0: 
                        missed += 1
                    missed_rate =  float(missed) / (curr_end-curr_start+1)
                    curr_height = pointcloud[curr_end,2] - pointcloud[curr_start,2]
                    if (missed > 10 and missed_rate > 0.3) or curr_height > max_curb_height or conti[i]:
                        break
                    if edge_end[i] and local_min_index[i]:
                        if curr_height > min_curb_height and curr_height < max_curb_height:
                            c_list = np.vstack((c_list,[[curr_start, curr_end]]))
                            break
                    i += 1
                curr_start, curr_end, curr_height = 0, 0, 0
            i += 1
    t_loop = time.time() - t_now

    t_now = time.time()
    # add curb detection result
    curb_result = np.zeros(n,'float')
    for c in c_list:
        curb_result[c[0]:c[1]] = 1.
    curb_result[edge_start*local_min_index] = 0.4
    curb_result[edge_end*local_min_index] = 0.7

    # only choose the first curb detection result
    first_curb_result = np.zeros(n,'float')
    for c in c_list:
        first_curb_result[c[0]:c[1]] = 1.
        curbs = np.vstack((curbs, pointcloud[c[0]:c[1]+1,0:5]))
        break

    # write detection results to pointcloud array
    pointcloud[edge_start,5] = 1.
    pointcloud[:,6] = edge_start
    pointcloud[:,7] = elevation
    pointcloud[:,8] = curb_result
    pointcloud[:,9] = first_curb_result
    
    t_result = time.time() - t_now
    return pointcloud, curbs, np.array([t_reorder*1000 ,t_elevation*1000,t_edge*1000,t_local_min*1000,t_conti*1000,t_loop*1000, t_result*1000])

def find_boundary_from_half_v01(pointcloud, half, k=8, n_result=5):
    """
    Detect and return boundary points index from single laser group at left or right side of vehicle
    with elevation filter / direction change filter

    @param pointcloud: input pointcloud (left or right & front in CW order) 
    @type: numpy array with shape (n, 5+n_result)
    @return: index of boundary points 
    @rtype: numpy array with shape (n, 5+n_result)
    """
    
    # parameters
    curbs = np.empty((0,5),'float') # np array for detection result of curb points
    c_list = np.empty((0,2),'int')
    curr_start, curr_end, curr_height = 0, 0, 0
    
    # return if not enough points
    n =  pointcloud.shape[0]
    if n - 2 * k < 0:
        return pointcloud, curbs, np.array([0. ,0., 0., 0., 0., 0., 0.])

    t_now = time.time()
    # reorder the points
    pointcloud = reorder_pointcloud(pointcloud)
    t_reorder = time.time() - t_now
    
    t_now = time.time()
    # elevation filter
    elevation = elevation_filter(pointcloud, half)
    t_elevation = time.time() - t_now
    
    t_now = time.time()
    # possible edge start and end from elevation info
    edge_start, edge_end = edge_filter_from_elevation(pointcloud, elevation, half)
    t_edge = time.time() - t_now

    t_now = time.time()
    # local min of direction change
    local_min_index = local_min_of_direction_change(pointcloud, half)
    t_local_min = time.time() - t_now
    
    t_now = time.time()
    # continuous filter
    conti = continuous_filter(pointcloud)
    t_conti = time.time() - t_now

    t_now = time.time()
    # main loop
    if half == 'left':
        i = n-1
        found = False
        while i >= 0:
            if local_min_index[i] and edge_start[i]:
                curr_start = i
                missed = 0
                while i-1 >= 0:
                    if local_min_index[i] and edge_start[i] and curr_height < min_curb_height:
                        curr_start = i
                    curr_end = i
                    if elevation[i] == 0: 
                        missed += 1
                    missed_rate =  float(missed) / (curr_start-curr_end+1)
                    curr_height = pointcloud[curr_end,2] - pointcloud[curr_start,2]
                    if (missed > 10 and missed_rate > 0.3): # might need to apply different check for "boundary"
                        break
                    if curr_height >  0.05 and edge_end[i]:
                        c_list = np.vstack((c_list,[[curr_end, curr_start]]))
                        found = True
                        break
                    if curr_height >  0.1:
                        c_list = np.vstack((c_list,[[curr_end, curr_start]]))
                        found = True
                        break
                    i -= 1
            i -= 1
            if found: break
    else:
        i = 0
        found = False
        while i < n:
            if local_min_index[i] and edge_start[i]:
                curr_start = i
                missed = 0
                while i+1 < n:
                    if local_min_index[i] and edge_start[i] and curr_height < min_curb_height:
                        curr_start = i
                    curr_end = i
                    if elevation[i] == 0: 
                        missed += 1
                    missed_rate =  float(missed) / (curr_end-curr_start+1)
                    curr_height = pointcloud[curr_end,2] - pointcloud[curr_start,2]
                    if (missed > 10 and missed_rate > 0.3):
                        break
                    if curr_height >  0.05 and edge_end[i]:
                        c_list = np.vstack((c_list,[[curr_start, curr_end]]))
                        found = True
                        break
                    if curr_height >  0.1:
                        c_list = np.vstack((c_list,[[curr_start, curr_end]]))
                        found = True
                        break
                    i += 1
                curr_start, curr_end, curr_height = 0, 0, 0
            i += 1
            if found: break
    t_loop = time.time() - t_now

    t_now = time.time()
    # add boundary detection result
    curb_result = np.zeros(n,'float')
    for c in c_list:
        curb_result[c[0]:c[1]] = 1.
    curb_result[edge_start*local_min_index] = 0.4
    curb_result[edge_end*local_min_index] = 0.7

    # only choose the first curb detection result
    # first_curb_result = np.zeros(n,'float')
    for c in c_list:
        curbs = np.vstack((curbs, pointcloud[c[0]:c[1]+1,0:5]))

    # write detection results to pointcloud array
    pointcloud[edge_start,5] = 1.
    pointcloud[:,6] = conti
    pointcloud[:,7] = elevation
    pointcloud[:,8] = curb_result
    
    # test obstacles
    obstacle_extraction(pointcloud, conti, half)

    t_result = time.time() - t_now
    return pointcloud, curbs, np.array([t_reorder*1000 ,t_elevation*1000,t_edge*1000,t_local_min*1000,t_conti*1000,t_loop*1000, t_result*1000])

def find_boundary_from_half_multiprocess(ring, half, k=8, n_result=5):
    """
    Detect and return boundary points index from single laser group at left or right side of vehicle
    with elevation filter / direction change filter

    @param pointcloud: input pointcloud (left or right & front in CW order) 
    @type: numpy array with shape (n, 5+n_result)
    @return: index of boundary points 
    @rtype: numpy array with shape (n, 5+n_result)
    """
    global shared_array, range_idx 
    idx = ring * 2
    if half == 'right': idx += 1
    st, ed = range_idx[idx]

    # parameters
    c_list = np.empty((0,2),'int')
    curr_start, curr_end, curr_height = 0, 0, 0
    
    # return if not enough points
    n = ed - st
    if n - 2 * k < 0:
        return

    # elevation filter
    elevation = elevation_filter(shared_array[st:ed], half)
    
    # possible edge start and end from elevation info
    edge_start, edge_end = edge_filter_from_elevation(shared_array[st:ed], elevation, half)

    # local min of direction change
    local_min_index = local_min_of_direction_change(shared_array[st:ed], half)
    
    # continuous filter
    conti = continuous_filter(shared_array[st:ed])

    # main loop
    if half == 'left':
        i = n-1
        found = False
        while i >= 0:
            if local_min_index[i] and edge_start[i]:
                curr_start = i
                missed = 0
                while i-1 >= 0:
                    if local_min_index[i] and edge_start[i] and curr_height < min_curb_height:
                        curr_start = i
                    curr_end = i
                    if elevation[i] == 0: 
                        missed += 1
                    missed_rate =  float(missed) / (curr_start-curr_end+1)
                    curr_height = shared_array[st+curr_end,2] - shared_array[st+curr_start,2]
                    if (missed > 10 and missed_rate > 0.3): # might need to apply different check for "boundary"
                        break
                    if curr_height >  0.05 and edge_end[i]:
                        c_list = np.vstack((c_list,[[curr_end, curr_start]]))
                        found = True
                        break
                    if curr_height >  0.1:
                        c_list = np.vstack((c_list,[[curr_end, curr_start]]))
                        found = True
                        break
                    i -= 1
            i -= 1
            if found: break
    else:
        i = 0
        found = False
        while i < n:
            if local_min_index[i] and edge_start[i]:
                curr_start = i
                missed = 0
                while i+1 < n:
                    if local_min_index[i] and edge_start[i] and curr_height < min_curb_height:
                        curr_start = i
                    curr_end = i
                    if elevation[i] == 0: 
                        missed += 1
                    missed_rate =  float(missed) / (curr_end-curr_start+1)
                    curr_height = shared_array[st+curr_end,2] - shared_array[st+curr_start,2]
                    if (missed > 10 and missed_rate > 0.3):
                        break
                    if curr_height >  0.05 and edge_end[i]:
                        c_list = np.vstack((c_list,[[curr_start, curr_end]]))
                        found = True
                        break
                    if curr_height >  0.1:
                        c_list = np.vstack((c_list,[[curr_start, curr_end]]))
                        found = True
                        break
                    i += 1
                curr_start, curr_end, curr_height = 0, 0, 0
            i += 1
            if found: break
    
    # add boundary detection result
    curb_result = np.zeros(n,'float')
    for c in c_list:
        curb_result[c[0]:c[1]] = 1.
    curb_result[edge_start*local_min_index] = 0.4
    curb_result[edge_end*local_min_index] = 0.7

    # write detection results to shared_array array
    # shared_array[edge_start,5] = 1.
    shared_array[st:ed,6] = conti
    shared_array[st:ed,7] = elevation
    shared_array[st:ed,8] = curb_result
    
    # test obstacles
    obstacle_extraction(shared_array[st:ed], conti, half)

def curb_detection_v2(msg, config, rot, height):
    """
    Detect and return ros message with additional "curb" information  
    Version two

    @param msg: input ros message
    @type: pointcloud2
    @param config: input config of the lidar sensor  
    @type: string
    @param rot: input rotation matrix
    @type: numpy array with shape (3, 3)
    @param height: input translation value in z direction
    @type: float
    @return: output ros message 
    @rtype: ros pointcloud2 message
    """
    pointcloud = get_pointcloud_from_msg(msg)
    if config == 'tilted': 
        pointcloud = rotate_pc(pointcloud, rot)
        pointcloud = translate_z_pc(pointcloud, height)
    else:
        pointcloud = translate_z_pc(pointcloud, height-0.2)
    pointcloud = max_height_filter(pointcloud, .45)
    pointcloud = FOV_positive_x_filter(pointcloud)

    pc_rearrange, index = rearrange_pointcloud_by_ring(pointcloud)
    
    # get pointcloud list
    pointcloud_list = get_pointcloud_list_by_ring_from_pointcloud(pointcloud)

    pc_data = np.empty((0,6),'float') 
    for i in range(16):
        pc_l = find_edge_from_left_half_v02(pointcloud_list[i]['left'])
        pc_r = find_edge_from_right_half_v02(pointcloud_list[i]['right'])
        pc_i = np.vstack((pc_l, pc_r))
        pc_data = np.vstack((pc_data, pc_i))

    return pc2_message(msg, pc_data)

def curb_detection_v3(pointcloud, config, rot, height, msg=None, n_result=5):
    """
    Detect and return ros message with additional "curb" information  
    Version three

    @param pointcloud: input pointcloud from realtime or rosbag msg
    @type: numpy array
    @param config: input config of the lidar sensor  
    @type: string
    @param rot: input rotation matrix
    @type: numpy array with shape (3, 3)
    @param height: input translation value in z direction
    @type: float
    @param msg: input ros message
    @type: pointcloud2
    @return: output ros message / pc_data, line_model_left, line_model_right
    @rtype: ros pointcloud2 message / numpy array with shape (n, 5+n_result),line model from skimage
    """
    start_time = time.time()
    if config == 'tilted': 
        pointcloud = rotate_pc(pointcloud, rot)
        pointcloud = translate_z_pc(pointcloud, height)
    else:
        pointcloud = translate_z_pc(pointcloud, height-0.2)
    pointcloud = max_height_filter(pointcloud, .45)
    pointcloud = FOV_positive_x_filter(pointcloud)

    t_now = time.time()
    pc_rearrange, index = rearrange_pointcloud_by_ring(pointcloud)
    t_rearrange = time.time() - t_now
    
    t_now = time.time()
    # get pointcloud list
    pointcloud_list = get_pointcloud_list_by_ring_from_pointcloud(pointcloud)
    t_to_list = time.time() - t_now

    # (x, y, z, index, ring)
    curbs_l = np.empty((0,5),'float') 
    curbs_r = np.empty((0,5),'float') 
    tt = np.zeros(7,'float')

    t_now = time.time()
    pc_data = np.empty((0,5+n_result),'float') 
    for i in range(16):
        # curr_index = pc_data.shape[0]
        pc_l, curb_l, t_l = find_curb_from_half_v031(pointcloud_list[i]['left'],'left')
        pc_r, curb_r, t_r = find_curb_from_half_v031(pointcloud_list[i]['right'],'right')
        pc_i = np.vstack((pc_l, pc_r))
        pc_data = np.vstack((pc_data, pc_i))
        curbs_l = np.vstack((curbs_l, curb_l))
        curbs_r = np.vstack((curbs_r, curb_r))
        tt += (t_l + t_r)
    t_detection = time.time() - t_now

    t_now = time.time()
    # ransac
    model_ransac_left, model_ransac_right = None, None
    if curbs_l.shape[0] > 0:
        model_ransac_left, inliers = ransac(curbs_l[:,:2], LineModelND, min_samples=2, residual_threshold=1, max_trials=100)
    if curbs_r.shape[0] > 0:
        model_ransac_right, inliers = ransac(curbs_r[:,:2], LineModelND, min_samples=2, residual_threshold=1, max_trials=100)
    t_ransac = time.time() - t_now

    if debug_print:
        print(tt)
        print(t_rearrange*1000, "ms, ", t_to_list*1000, "ms, ", t_detection*1000, "ms ", "ms ", t_ransac*1000, "ms ", (time.time()-start_time)*1000, "ms")

    # realtime option    
    if msg == None:
        return pc_data, model_ransac_left, model_ransac_right
    # rosbag option    
    else:
        line_x = np.arange(0, 25, 0.1)
        if model_ransac_left != None:
            line_y_robust_left = model_ransac_left.predict_y(line_x)
            point_line_l = np.zeros((250,5+n_result),'float') 
            point_line_l[:,0] = line_x
            point_line_l[:,1] = line_y_robust_left
            point_line_l[:,3] = 10
            point_line_l[:,8] = .5
            point_line_l[:,9] = .5
            pc_data = np.vstack((pc_data, point_line_l))
        if model_ransac_right != None:
            line_y_robust_right = model_ransac_right.predict_y(line_x)
            point_line_r = np.zeros((250,5+n_result),'float') 
            point_line_r[:,0] = line_x
            point_line_r[:,1] = line_y_robust_right
            point_line_r[:,3] = 10
            point_line_r[:,8] = .5
            point_line_r[:,9] = .5
            pc_data = np.vstack((pc_data, point_line_r))
        return pc2_message(msg, pc_data)

def boundary_detection_v1(pointcloud, config, rot, height, msg=None, n_result=5):
    """
    Detect and return ros message with additional "boundary" information  
    Version one

    @param pointcloud: input pointcloud from realtime or rosbag msg
    @type: numpy array
    @param config: input config of the lidar sensor  
    @type: string
    @param rot: input rotation matrix
    @type: numpy array with shape (3, 3)
    @param height: input translation value in z direction
    @type: float
    @param msg: input ros message
    @type: pointcloud2
    @return: output ros message / pc_data, line_model_left, line_model_right
    @rtype: ros pointcloud2 message / numpy array with shape (n, 5+n_result),line model from skimage
    """
    start_time = time.time()
    if config == 'tilted': 
        pointcloud = rotate_pc(pointcloud, rot)
        pointcloud = translate_z_pc(pointcloud, height)
    else:
        pointcloud = translate_z_pc(pointcloud, height-0.2)
    # pointcloud = max_height_filter(pointcloud, .45)
    pointcloud = max_height_filter(pointcloud, 1.)
    pointcloud = FOV_positive_x_filter(pointcloud)

    t_now = time.time()
    # pc_rearrange, index = rearrange_pointcloud_by_ring(pointcloud)
    t_rearrange = time.time() - t_now
    
    t_now = time.time()
    # get pointcloud list
    pointcloud_list = get_pointcloud_list_by_ring_from_pointcloud(pointcloud)
    t_to_list = time.time() - t_now

    # (x, y, z, index, ring)
    curbs_l = np.empty((0,5),'float') 
    curbs_r = np.empty((0,5),'float') 
    tt = np.zeros(7,'float')

    t_now = time.time()

    pc_data = np.zeros((0,5+n_result),'float') 
    for i in range(16):
        pc_l, curb_l, t_l = find_boundary_from_half_v01(pointcloud_list[i]['left'],'left')
        pc_r, curb_r, t_r = find_boundary_from_half_v01(pointcloud_list[i]['right'],'right')
        pc_i = np.vstack((pc_l, pc_r))
        pc_data = np.vstack((pc_data, pc_i))
        curbs_l = np.vstack((curbs_l, curb_l))
        curbs_r = np.vstack((curbs_r, curb_r))
        tt += (t_l + t_r)
    t_detection = time.time() - t_now

    t_now = time.time()
    # ransac
    model_ransac_left, model_ransac_right = None, None
    if curbs_l.shape[0] > 0:
        model_ransac_left, inliers = ransac(curbs_l[:,:2], LineModelND, min_samples=2, residual_threshold=1, max_trials=100)
    if curbs_r.shape[0] > 0:
        model_ransac_right, inliers = ransac(curbs_r[:,:2], LineModelND, min_samples=2, residual_threshold=1, max_trials=100)
    t_ransac = time.time() - t_now

    # poly fit 
    model_l, model_r = None, None
    if curbs_l.shape[0] > 0:
        poly_l = np.polyfit(curbs_l[:,0], curbs_l[:,1], 5)  
        model_l = np.poly1d(poly_l)    
    if curbs_r.shape[0] > 0:
        poly_r = np.polyfit(curbs_r[:,0], curbs_r[:,1], 5)      
        model_r = np.poly1d(poly_r)    

    if debug_print:
        print(tt)
        print(t_rearrange*1000, "ms, ", t_to_list*1000, "ms, ", t_detection*1000, "ms ", "ms ", t_ransac*1000, "ms ", (time.time()-start_time)*1000, "ms")

    # realtime option    
    if msg == None:
        return pc_data, model_ransac_left, model_ransac_right
    # rosbag option    
    else:
        max_dis = 15
        line_x = np.arange(0, max_dis, 0.1)
        if model_ransac_left != None:
            line_y_robust_left = model_ransac_left.predict_y(line_x)
            point_line_l = np.zeros((max_dis*10,5+n_result),'float') 
            point_line_l[:,0] = line_x
            point_line_l[:,1] = line_y_robust_left
            point_line_l[:,3] = 10
            point_line_l[:,8] = .5
            point_line_l[:,9] = .5
            pc_data = np.vstack((pc_data, point_line_l))
        if model_ransac_right != None:
            line_y_robust_right = model_ransac_right.predict_y(line_x)
            point_line_r = np.zeros((max_dis*10,5+n_result),'float') 
            point_line_r[:,0] = line_x
            point_line_r[:,1] = line_y_robust_right
            point_line_r[:,3] = 10
            point_line_r[:,8] = .5
            point_line_r[:,9] = .5
            pc_data = np.vstack((pc_data, point_line_r))

        return pc2_message(msg, pc_data)

def find_boundary_from_scan(i, num_of_process):
    num_per_process = 16 // num_of_process
    for j in range (num_per_process*i, num_per_process*(i+1)):
        find_boundary_from_half_multiprocess(j,'left')
        find_boundary_from_half_multiprocess(j,'right')

def boundary_detection_v1_multiprocess(pointcloud, config, rot, height, msg=None, n_result=5):
    """
    Detect and return ros message with additional "boundary" information  
    Version one

    @param pointcloud: input pointcloud from realtime or rosbag msg
    @type: numpy array
    @param config: input config of the lidar sensor  
    @type: string
    @param rot: input rotation matrix
    @type: numpy array with shape (3, 3)
    @param height: input translation value in z direction
    @type: float
    @param msg: input ros message
    @type: pointcloud2
    @return: output ros message / pc_data, line_model_left, line_model_right
    @rtype: ros pointcloud2 message / numpy array with shape (n, 5+n_result),line model from skimage
    """
    start_time = time.time()
    if config == 'tilted': 
        pointcloud = rotate_pc(pointcloud, rot)
        pointcloud = translate_z_pc(pointcloud, height)
    else:
        pointcloud = translate_z_pc(pointcloud, height-0.2)
    # pointcloud = max_height_filter(pointcloud, .45)
    pointcloud = max_height_filter(pointcloud, 1.)
    pointcloud = FOV_positive_x_filter(pointcloud)

    t_now = time.time()
    # pc_rearrange, index = rearrange_pointcloud_by_ring(pointcloud)
    t_rearrange = time.time() - t_now
    
    t_now = time.time()
    # get pointcloud list
    # pointcloud_list = get_pointcloud_list_by_ring_from_pointcloud(pointcloud)
    t_to_list = time.time() - t_now

    tt = np.zeros(7,'float')

    t_now = time.time()

    pointcloud_re, idx = get_rearranged_pointcloud(pointcloud)
    result_col = np.zeros((pointcloud_re.shape[0], 5),'float')
    pointcloud_re = np.hstack((pointcloud_re, result_col))

    global range_idx
    global shared_array 
    range_idx = idx

    # create shared array *** size of c_double and np.float64 ***
    m, n =  pointcloud_re.shape
    shared_arr_based = multiprocessing.RawArray(ctypes.c_double, m*n)
    shared_arr = np.frombuffer(shared_arr_based, dtype=np.float64).reshape(pointcloud_re.shape)
    np.copyto(shared_arr, pointcloud_re)
    shared_array = shared_arr

    # process list
    processes = []
    num_of_process = 4
    for i in range(num_of_process):
        processes.append(multiprocessing.Process(target = find_boundary_from_scan, args= (i, num_of_process,)))
        processes[i].start()
    t_detection = time.time() - t_now

    for i in range(num_of_process):
        processes[i].join()

    # get left and right list
    curbs_l = shared_array[np.bitwise_and(shared_array[:,1] > 0., shared_array[:,8] == 1.)]
    curbs_r = shared_array[np.bitwise_and(shared_array[:,1] <= 0., shared_array[:,8] == 1.)]

    t_now = time.time()
    # ransac
    model_ransac_left, model_ransac_right = None, None
    if curbs_l.shape[0] > 0:
        model_ransac_left, inliers = ransac(curbs_l[:,:2], LineModelND, min_samples=2, residual_threshold=1, max_trials=100)
    if curbs_r.shape[0] > 0:
        model_ransac_right, inliers = ransac(curbs_r[:,:2], LineModelND, min_samples=2, residual_threshold=1, max_trials=100)
    t_ransac = time.time() - t_now

    if debug_print:
        print(tt)
        print (t_rearrange*1000, "ms, ", t_to_list*1000, "ms, ", t_detection*1000, "ms ", "ms ", t_ransac*1000, "ms ", (time.time()-start_time)*1000, "ms")

    # realtime option    
    if msg == None:
        return pc_data, model_ransac_left, model_ransac_right
    # rosbag option    
    else:
        max_dis = 15
        line_x = np.arange(0, max_dis, 0.1)
        if model_ransac_left != None:
            line_y_robust_left = model_ransac_left.predict_y(line_x)
            point_line_l = np.zeros((max_dis*10,5+n_result),'float') 
            point_line_l[:,0] = line_x
            point_line_l[:,1] = line_y_robust_left
            point_line_l[:,3] = 10
            point_line_l[:,8] = .5
            point_line_l[:,9] = .5
            shared_array = np.vstack((shared_array, point_line_l))
        if model_ransac_right != None:
            line_y_robust_right = model_ransac_right.predict_y(line_x)
            point_line_r = np.zeros((max_dis*10,5+n_result),'float') 
            point_line_r[:,0] = line_x
            point_line_r[:,1] = line_y_robust_right
            point_line_r[:,3] = 10
            point_line_r[:,8] = .5
            point_line_r[:,9] = .5
            shared_array = np.vstack((shared_array, point_line_r))

        return pc2_message(msg, shared_array)

def run_and_save_to_bin(data_name, data, config, detection_type, visualize=False, tilted_angle=19.2, height=1.195):
    if config not in ['horizontal', 'tilted']:
        print('Invalid config input, should be horizontal or tilted')
        return

    idx = 0
    for topic_1, msg_1, t_1 in data.topic_1:
        filename = 'test/' + str(idx) + '.bin'
        f = open(filename, mode='wb')
        print('frame', idx, '/', lidar_data.len_1)
        pointcloud = get_pointcloud_from_msg(msg_1)
        np.asarray(pointcloud, dtype=np.float32).tofile(f)
        idx += 1
    output_bag.close()

def run_detection_and_save(data_name, data, config, detection_type, visualize=False, tilted_angle=19.2, height=1.195):
    """
    Run curb detection algorithm throught all messages in data and store as new rosbag file

    @param data_name: input name of the rosbag file
    @type: string
    @param data: input lidar data
    @type: RosbagParser object
    @param config: input config of the lidar sensor  
    @type: string
    @param visualize: set visualize to True to visualize the result in open3D   
    @type: boolean
    @param tilted_angle: input tilted angle of lidar sensor in degree
    @type: float
    @param height: input translation value in z direction
    @type: float
    """
    if config not in ['horizontal', 'tilted']:
        print('Invalid config input, should be horizontal or tilted')
        return
    if detection_type == 'curb':    
        bag_name = result_path + data_name.split('/')[-1].split('.')[0] + '_curb.bag'
    elif detection_type == 'boundary':
        bag_name = result_path + data_name.split('/')[-1].split('.')[0] + '_boundary.bag'
    output_bag = rosbag.Bag(bag_name, 'w')

    if visualize:
        # initialize visualizer
        vis = open3d.Visualizer()
        vis.create_window(window_name='point cloud', width=1280, height=960)
        pcd = open3d.PointCloud()
        ctr = vis.get_view_control()

    # /image_raw
    for topic_0, msg_0, t_0 in data.topic_0:
        output_bag.write(topic_0, msg_0, t=t_0)

    rot = rotation_matrix(tilted_angle)
    # /points_raw
    avg_time = 0
    idx = 0
    for topic_1, msg_1, t_1 in data.topic_1:
        print('frame', idx, '/', lidar_data.len_1)
        start_time = time.time()
        pointcloud = get_pointcloud_from_msg(msg_1)
        msg_1_processed = boundary_detection_v1_multiprocess(pointcloud, config, rot, height, msg_1) # run curb detection algorithm 
        # msg_1_processed = boundary_detection_v1(pointcloud, config, rot, height, msg_1) # run curb detection algorithm 
        process_time = (time.time() - start_time)* 1000
        print(process_time, "ms")
        avg_time += process_time 
        output_bag.write(topic_1, msg_1_processed, t=t_1)
        pointcloud_p = get_pointcloud_from_msg(msg_1_processed)
        if visualize:
            color_map = get_color_from_curb(pointcloud_p) 
            # visualizing lidar points in camera coordinates
            if idx == 0:
                pcd.points = open3d.Vector3dVector(pointcloud_p[:,:3])
                vis.add_geometry(pcd)
            update_vis(vis, pcd, pointcloud_p[:,:3], color_map)
        idx += 1
    print("Average time:", avg_time / lidar_data.len_1)
    if visualize:
        vis.destroy_window()
    output_bag.close()

def run_detection_and_display(path, config, tilted_angle=19.2, height=1.195):
    """
    Run curb detection algorithm in real time
    
    @param path: path to the bin file of the point cloud data
    @type: string
    @param config: input config of the lidar sensor  
    @type: string
    @param tilted_angle: input tilted angle of lidar sensor in degree
    @type: float
    @param height: input translation value in z direction
    @type: float
    """
    if config not in ['horizontal', 'tilted']:
        print('Invalid config input, should be horizontal or tilted')
        return
    rot = rotation_matrix(tilted_angle)
    
    # initialize visualizer
    vis = open3d.Visualizer()
    vis.create_window(window_name='point cloud', width=1280, height=960)
    pcd = open3d.PointCloud()
    ctr = vis.get_view_control()

    # draw coodinate axis at (0, 0, -0.9)
    add_origin_axis(vis)

    idx = 0
    while True:
        print('frame', idx)
        pointcloud = np.fromfile(path, dtype=np.float32).reshape(-1, 5)
        n = pointcloud.shape[0]
        pointcloud_p, left_model, right_model = curb_detection_v3(pointcloud, config, rot, height) # run curb detection algorithm 

        color_map = get_color_from_curb(pointcloud_p) 
        # visualizing lidar points in camera coordinates
        if idx == 0:
            pcd.points = open3d.Vector3dVector(pointcloud_p[:,:3])
            vis.add_geometry(pcd)
        update_vis(vis, pcd, pointcloud_p[:,:3], color_map)
        idx += 1
    vis.destroy_window()

topics = ['/camera/image_raw', '/points_raw']
path_0424 = '/home/rtml/LiDAR_camera_calibration_work/data/data_bag/20190424_pointgrey/'
path_0517 = '/home/rtml/LiDAR_camera_calibration_work/data/data_bag/20190517_pointgrey/'
result_path = '/home/rtml/Lidar_curb_detection/source/lidar_based/results/'

parser = argparse.ArgumentParser(description='Run with either \'rosbag\' or \'realtime\' option')
parser.add_argument('source', type=str, help='From rosbag file or from bin file in real time')
args = parser.parse_args()
if __name__ == '__main__':
    if args.source not in ['rosbag', 'realtime']:
        print('Invalid argument, should be \'rosbag\' or \'realtime\'')
        sys.exit()
    
    # rosbag option: read from source rosbag file and save the result rosbag at result_path
    if args.source == 'rosbag': 
        data_path = data_path_loader(path_0517)
        # change the number to read different rosbag file
        # tilted: 0 to 9 
        # horizontal: 0 to 5 
        data_name = data_path['tilted'][23]
        print(data_name)
        lidar_data = RosbagParser(data_name, topics)
        # set "visualize = True" to visualize the result in open3D
        # run_detection_and_save(data_name, lidar_data, 'tilted', 'boundary', visualize=False, tilted_angle=15., height=1.125)
        run_and_save_to_bin(data_name, lidar_data, 'tilted', 'boundary', visualize=False, tilted_angle=15., height=1.125)
    
    # realtime option: continuously read from bin file at data_path and visualize through open3D 
    else:
        data_path = '/home/rtml/Lidar_curb_detection/source/lidar_based/curb_detection/image.bin'
        run_detection_and_display(data_path, 'tilted')

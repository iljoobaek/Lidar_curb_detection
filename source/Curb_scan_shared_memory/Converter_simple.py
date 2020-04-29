# Converting the ply files into the bin files

import sys 
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
print("removed ros")

import pyntcloud
import numpy as np
import glob
import pdb
import os
import json
import matplotlib.pyplot as plt
import argoverse
import shutil
from tqdm import tqdm
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
import argoverse.visualization.visualization_utils as viz_util
from scipy.spatial.transform import Rotation

'''
 To get the internsity and ring information from argoverse data just attach this code to plyfile.py
 inside the api folder

def convert_listed_path(file_path):
    data = pyntcloud.PyntCloud.from_file(os.fspath(file_path))
    x = np.array(data.points.x)[:, np.newaxis]
    y = np.array(data.points.y)[:, np.newaxis]
    z = np.array(data.points.z)[:, np.newaxis]
    i = np.array(data.points.intensity)[:, np.newaxis]
    ring = np.array(data.points.laser_number)[:, np.newaxis]
    
    return np.concatenate((x, y, z, i, ring), axis=1)

if __name__ == "__main__":
    pointcloud_paths = os.path.join("/home/droid/manoj_work/data/argoverse/scene1/lidar", "*")
    pcd_list = glob.glob(pointcloud_paths)
    save_folder = os.path.join("/home/droid/manoj_work/data/argoverse/", "argo_sample")
    if(not os.path.exists(save_folder)): os.makedirs(save_folder)
    
    for i, each in enumerate(pcd_list):
        save_path = os.path.join(save_folder, "{0:010d}".format(i)+".bin")
        pointcloud = convert_listed_path(each)
        np.array(pointcloud).tofile(save_path)
        print("Saved ", each) 

'''


def get_objects_from_label(label_file):
    # Opens a label file, and passes the object to Object3d object, Read the json GT labels
    
    f = open(label_file)
    label_data = json.load(f) 
    objects = [object3d.Object3d(data) for data in label_data]
    return objects
    
def check_make_(save_dir, display=True):
    if not os.path.exists(save_dir):
        if display:
            print("Making folder ", save_dir)
        os.makedirs(save_dir)
    else:
        if display:
            print("Folder present ", save_dir)
        else:
            pass

def dump(calib_file, calib):
    with open(calib_file, 'w') as outfile:
        json.dump(calib, outfile, ensure_ascii=False, indent=4)

def dump_img(file, img):
    plt.imsave(file, img , format='png')
        
def dump_pcd(file, pcd):
    pcd.tofile(file)
    
def cpy(source, destination):
    shutil.copy(source, destination) 
       
    
    
if __name__=="__main__":
    '''
    Data-format:
    data/
        kitti/
        Argoverse/
            val/
            train/
                log_id_1/
                    image_01/
                    velodyne_points/
                        data/
                        timestamps.txt/
                log_id_2/
                    ...
    '''
    cwd = os.getcwd()
    base_root = "/home/droid/manoj_work"
    root_dir = os.path.join(base_root, "data", "raw_Argoverse") + "/"
    save_h5_root = os.path.join(base_root, "data", "Argoverse") + "/"
    
    dataset = ['train', 'val', 'test']
    folders = [os.path.join(root_dir, folder) for folder in ['train', 'val', 'test']]
    
    choice = 0 #args.dataset_choice
    folder_choice = folders[choice]
    dataset_choice = dataset[choice]
    
    root_dir_choice = os.path.join(root_dir, dataset_choice)
    save_dir = os.path.join(save_h5_root, dataset_choice)
    check_make_(save_dir)
    
    split = folder_choice[len(os.path.dirname(folder_choice))+1:]
    is_test = (split == 'test')
    lidar_pathlist = []
    label_pathlist = []
    actual_idx_list = []
    calib_objects= []
    logidx_to_count_map= {}
    log_to_count_map= {}

    print("____________SPLIT IS : {} ______________".format(split))
    if split == 'train' or split == 'val':
        
        imageset_dir = os.path.join(root_dir,split)
        splitname = lambda x: [x[len(imageset_dir+"/"):-4].split("/")[0], x[len(imageset_dir+"/"):-4].split("/")[2].split("_")[1]]
        
        data_loader = ArgoverseTrackingLoader(os.path.join(root_dir,split))
        camera = data_loader.CAMERA_LIST[0]
        print("Chosen camera is : ", camera)
        log_list = data_loader.log_list
        path_count = 0
        for log_id, log in enumerate(log_list):
            
            log_folder = os.path.join(save_dir, "{0:010d}".format(log_id))
            check_make_(log_folder)
            pcd_folder = os.path.join(log_folder, "velodyne_points", "data")
            check_make_(pcd_folder)
            oxts_folder = os.path.join(log_folder, "oxts")
            check_make_(oxts_folder)
            img_folder = os.path.join(log_folder, "image_01", "data")
            check_make_(img_folder)
            img_box_folder = os.path.join(log_folder, "imagebox_01", "data")
            check_make_(img_box_folder)
            calib = data_loader.get_calibration(camera)
            calib_src = os.path.join(root_dir_choice, log, "vehicle_calibration_info.json")
            oxts_root_src = os.path.join(root_dir_choice, log, "poses")
            cpy(calib_src, os.path.join(log_folder, "calib_{0:010d}".format(log_id)+".json"))
            
            argoverse_data = data_loader.get(log)
            lidar_lst = data_loader.get(log).lidar_list
            lidar_pathlist.extend(lidar_lst)
            label_pathlist.extend(data_loader.get(log).label_list)
            assert len(lidar_pathlist) == len(label_pathlist)
                
            for idx, each_path in enumerate(tqdm(lidar_lst)):
                
                ide = log_id + idx
                img = argoverse_data.get_image_sync(ide, camera = camera)
                objects = argoverse_data.get_label_object(ide)
                img_vis = viz_util.show_image_with_boxes(img, objects, calib)
                lidar_pts = argoverse_data.get_lidar(ide)
                
		img_file = os.path.join(img_folder, "{0:010d}".format(ide)+".json")
		img_box_file = os.path.join(img_box_folder, "{0:010d}".format(ide)+".json")
		objects_file = os.path.join(oxts_folder, "{0:010d}".format(ide)+".json")

                dump_img(os.path.join(img_folder, "{0:010d}".format(ide)+".png"), img)
                dump_img(os.path.join(img_box_folder, "{0:010d}".format(ide)+".png"), img_vis)
		dump(os.path.join(oxts_folder, "{0:010d}".format(ide)+".json"), objects)
                dump_pcd(os.path.join(pcd_folder, "{0:010d}".format(ide)+".bin"), lidar_pts)
             
            actual_idx_list.extend([splitname(each) for each in lidar_lst])
            idx_list = np.arange(path_count, path_count + len(lidar_lst))
            logidx_to_count_map[log_id] = idx_list
            log_to_count_map[log] = idx_list
            path_count+=len(lidar_lst)
        
        dump(os.path.join(save_dir, "logidx_to_count_map.json"), logidx_to_count_map)
        dump(os.path.join(save_dir, "log_to_count_map.json"), log_to_count_map)


        
    

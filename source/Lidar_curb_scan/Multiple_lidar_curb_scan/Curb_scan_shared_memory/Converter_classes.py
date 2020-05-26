# Converting the ply files into the bin files

import pyntcloud
import numpy as np
import glob
import pdb
import os
import json

import argoverse
from argoverse.data_loading.argoverse_tracking_loader import ArgoverseTrackingLoader
from scipy.spatial.transform import Rotation

# def convert_listed_path(file_path):
    
#     data = pyntcloud.PyntCloud.from_file(os.fspath(file_path))
#     x = np.array(data.points.x)[:, np.newaxis]
#     y = np.array(data.points.y)[:, np.newaxis]
#     z = np.array(data.points.z)[:, np.newaxis]
#     i = np.array(data.points.intensity)[:, np.newaxis]
#     ring = np.array(data.points.laser_number)[:, np.newaxis]
    
#     return np.concatenate((x, y, z, i, ring), axis=1)

# if __name__ == "__main__":
#     pointcloud_paths = os.path.join("/home/droid/manoj_work/data/argoverse/scene1/lidar", "*")
#     pcd_list = glob.glob(pointcloud_paths)
#     save_folder = os.path.join("/home/droid/manoj_work/data/argoverse/", "argo_sample")
#     if(not os.path.exists(save_folder)): os.makedirs(save_folder)
    
#     for i, each in enumerate(pcd_list):
#         save_path = os.path.join(save_folder, "{0:010d}".format(i)+".bin")
#         pointcloud = convert_listed_path(each)
#         np.array(pointcloud).tofile(save_path)
# #         np.savetxt(save_path, pointcloud)
#         print("Saved ", each)

def get_calib_from_file(calib_file):
    with open(calib_file) as f:
        calib_data = json.load(f)
    
    P2 = np.array([[calib_data['camera_data_'][5]['value']['focal_length_x_px_'],0.0,calib_data['camera_data_'][5]['value'] ['focal_center_x_px_'],0.0],[0.0,calib_data['camera_data_'][5]['value']['focal_length_y_px_'],calib_data['camera_data_'][5]['value']['focal_center_y_px_'],0.0],[0.0,0.0,1.0,0.0]])

    R0 = np.eye(3)
    Tr_velo_to_cam = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0]])

    return {'P2': P2.reshape(3, 4),
            'P3': None,
            'R0': R0.reshape(3, 3),
            'Tr_velo2cam': Tr_velo_to_cam.reshape(3, 4)}


def get_objects_from_label(label_file):
    # Opens a label file, and passes the object to Object3d object, Read the json GT labels
    
    f = open(label_file)
    label_data = json.load(f) 
    objects = [object3d.Object3d(data) for data in label_data]
    return objects

def cls_type_to_id(cls_type):
    type_to_id = {'VEHICLE': 1, 'Pedestrian': 2, 'Cyclist': 3, 'Van': 4}
    if cls_type not in type_to_id.keys():
        return -1
    return type_to_id[cls_type]


class Object3d(object):
    def __init__(self, line):
        self.argo_to_kitti = np.array([[0, -1, 0],
                                       [0, 0, -1],
                                       [1, 0, 0]])
        
        label = line
        self.src = line
        self.cls_type = label['label_class']
        self.cls_id = cls_type_to_id(self.cls_type)
        
        self.trucation = 0.0
        self.occlusion = 0.0  
        self.alpha = np.arctan2(label['center']['z'],label['center']['x'])
        
        self.h = float(label['height'])
        self.w = float(label['width'])
        self.l = float(label['length'])
        self.pos_argo = np.array([float(label['center']['x']), float(label['center']['y']), float(label['center']['z'])], dtype=np.float32)
        
        #KITTI Frame
        self.pos = np.dot(self.argo_to_kitti,self.pos_argo)
        w,x,y,z = label['rotation']['w'],label['rotation']['x'],label['rotation']['y'],label['rotation']['z']
        self.q = np.array([x, y, z, w])       
        self.rot_mat_argo = Rotation.from_quat(self.q).as_dcm()
        
        
        self.ry = -Rotation.from_quat(self.q).as_euler('xyz')[-1] + np.pi/2.
        self.score = -1.0
        self.level_str = None
        self.level = self.get_obj_level()

    def get_obj_level(self):
        # Orginal : Assign level based on height of bounidng box in image, truncation, and occulusion value
        
        # Modified: Assign level based on distance from Origin of Lidar. Done
        distance = np.linalg.norm(self.pos)

        if distance <= 30.0:
            self.level_str = 'Easy'
            return 1  # Easy
        elif distance > 30.0 and distance <= 60.0:
            self.level_str = 'Moderate'
            return 2  # Moderate
        elif distance > 60 :
            self.level_str = 'Hard'
            return 3  # Hard
        else:
            self.level_str = 'UnKnown'
            return 4
        
    def generate_corners3d(self):
        """
        generate corners3d representation for this object
        :return corners_3d: (8, 3) corners of box3d in camera coord
        """
        l, h, w = self.l, self.h, self.w
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [h/2., h/2., h/2., h/2., -h/2., -h/2., -h/2., -h/2.]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        R = np.array([[np.cos(self.ry), 0, np.sin(self.ry)],
                      [0, 1, 0],
                      [-np.sin(self.ry), 0, np.cos(self.ry)]])
        corners3d = np.vstack([x_corners, y_corners, z_corners])  # (3, 8)
        corners3d = np.dot(R, corners3d).T
        corners3d = corners3d + self.pos
        return corners3d

    def to_bev_box2d(self, oblique=True, voxel_size=0.1):
        """
        :param bev_shape: (2) for bev shape (h, w), => (y_max, x_max) in image
        :param voxel_size: float, 0.1m
        :param oblique:
        :return: box2d (4, 2)/ (4) in image coordinate
        """
        if oblique:
            corners3d = self.generate_corners3d()
            xz_corners = corners3d[0:4, [0, 2]]
            box2d = np.zeros((4, 2), dtype=np.int32)
            box2d[:, 0] = ((xz_corners[:, 0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            box2d[:, 1] = Object3d.BEV_SHAPE[0] - 1 - ((xz_corners[:, 1] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            box2d[:, 0] = np.clip(box2d[:, 0], 0, Object3d.BEV_SHAPE[1])
            box2d[:, 1] = np.clip(box2d[:, 1], 0, Object3d.BEV_SHAPE[0])
        else:
            box2d = np.zeros(4, dtype=np.int32)
            # discrete_center = np.floor((self.pos / voxel_size)).astype(np.int32)
            cu = np.floor((self.pos[0] - Object3d.MIN_XZ[0]) / voxel_size).astype(np.int32)
            cv = Object3d.BEV_SHAPE[0] - 1 - ((self.pos[2] - Object3d.MIN_XZ[1]) / voxel_size).astype(np.int32)
            half_l, half_w = int(self.l / voxel_size / 2), int(self.w / voxel_size / 2)
            box2d[0], box2d[1] = cu - half_l, cv - half_w
            box2d[2], box2d[3] = cu + half_l, cv + half_w

        return box2d

    def to_str(self):
        print_str = '%s %.3f %.3f %.3f box2d: %s hwl: [%.3f %.3f %.3f] pos: %s ry: %.3f' \
                     % (self.cls_type, self.trucation, self.occlusion, self.alpha, self.box2d, self.h, self.w, self.l,
                        self.pos, self.ry)
        return print_str

    def to_kitti_format(self):
        kitti_str = '%s %.2f %d %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f' \
                    % (self.cls_type, self.trucation, int(self.occlusion), self.alpha, self.box2d[0], self.box2d[1],
                       self.box2d[2], self.box2d[3], self.h, self.w, self.l, self.pos[0], self.pos[1], self.pos[2],
                       self.ry)
        return kitti_str

class Calibration(object):
    def __init__(self, calib_file):
        if isinstance(calib_file, str):
            calib = get_calib_from_file(calib_file)
        else:
            calib = calib_file

        self.P2 = calib['P2']  # 3 x 4
        self.R0 = calib['R0']  # 3 x 3
        self.V2C = calib['Tr_velo2cam']  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def depthmap_to_rect(self, depth_map):
        """
        :param depth_map: (H, W), depth_map
        :return:
        """
        x_range = np.arange(0, depth_map.shape[1])
        y_range = np.arange(0, depth_map.shape[0])
        x_idxs, y_idxs = np.meshgrid(x_range, y_range)
        x_idxs, y_idxs = x_idxs.reshape(-1), y_idxs.reshape(-1)
        depth = depth_map[y_idxs, x_idxs]
        pts_rect = self.img_to_rect(x_idxs, y_idxs, depth)
        return pts_rect, x_idxs, y_idxs

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, self.P2.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

    def camera_dis_to_rect(self, u, v, d):
        """
        Can only process valid u, v, d, which means u, v can not beyond the image shape, reprojection error 0.02
        :param u: (N)
        :param v: (N)
        :param d: (N), the distance between camera and 3d points, d^2 = x^2 + y^2 + z^2
        :return:
        """
        assert self.fu == self.fv, '%.8f != %.8f' % (self.fu, self.fv)
        fd = np.sqrt((u - self.cu)**2 + (v - self.cv)**2 + self.fu**2)
        x = ((u - self.cu) * d) / fd + self.tx
        y = ((v - self.cv) * d) / fd + self.ty
        z = np.sqrt(d**2 - x**2 - y**2)
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)), axis=1)
        return pts_rect

class KittiDataset(torch_data.Dataset):
    def __init__(self, root_dir, split='train'):
        self.split = split
        is_test = (self.split == 'test')
        self.lidar_pathlist = []
        self.label_pathlist = []
        
        self.lidar_dir = os.path.join(root_dir)
        data_loader = ArgoverseTrackingLoader(os.path.join(root_dir))
        self.lidar_pathlist.extend(data_loader.lidar_list)
        self.label_pathlist.extend(data_loader.label_list)
        #self.calib_file = data_loader.calib_filename
        self.lidar_filename = [x.split('.')[0].rsplit('/',1)[1] for x in self.lidar_pathlist]
        
        assert len(self.lidar_pathlist) == len(self.label_pathlist)

        self.num_sample = len(self.lidar_pathlist)
        self.lidar_idx_list = ['%06d'%l for l in range(self.num_sample)]
        
        self.lidar_idx_table = dict(zip(self.lidar_idx_list, self.lidar_filename))
        self.argo_to_kitti = np.array([[0, -1, 0],
                                       [0, 0, -1],
                                       [1, 0, 0 ]])

        self.ground_removal = False
        
        self.lidar_dir = os.path.join('/data/')        
        self.label_dir = os.path.join('/data/')
        
    def get_lidar(self,idx):
        lidar_file = self.lidar_pathlist[idx]
        assert os.path.exists(lidar_file)
        
        
        data = PyntCloud.from_file(lidar_file)
        x = np.array(data.points.x)[:, np.newaxis]
        y = np.array(data.points.y)[:, np.newaxis]
        z = np.array(data.points.z)[:, np.newaxis]
        pts_lidar = np.concatenate([x,y,z], axis = 1)
        
        if self.ground_removal: 
            pts_lidar = gs.ground_segmentation(pts_lidar)
        
        pts_lidar = np.dot(self.argo_to_kitti,pts_lidar.T).T
        
        return pts_lidar
        
        
    def get_label(self,idx):
        
        label_file = self.label_pathlist[idx]
        assert os.path.exists(label_file)
        
        return get_objects_from_label(label_file)    

    def get_calib(self, idx):
        # Single Calibration File for All, One Calib to Rule them all
        assert os.path.exists(self.calib_file)
        return calibration.Calibration(self.calib_file)

    def get_image_shape(self, idx):
        return 1200, 1920, 3
    
    def get_image(self, idx):
        import cv2
        # cv2.setNumThreads(0)  # for solving deadlock when switching epoch
        img_file = os.path.join(self.image_dir, '%06d.png' % idx)
        assert os.path.exists(img_file)
        return cv2.imread(img_file)  # (H, W, 3) BGR mode

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError
    
    
if __name__=="__main__":
    
        
    
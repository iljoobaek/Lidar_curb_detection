#!/usr/bin/env python

# Evaluate the accuracy of curb detection

import os
import glob

class Eval:
    def __init__(self, data_paths, index_range):
        self.frames = 0
        self.gt_root_dir = data_paths['ground_truth_data']
        self.test_root_dir = data_paths['test_data']

    def getFileNames(self, idx):
        paths = {}
        paths['gt_left'] = os.path.join(self.gt_root_dir, 'gt_{:010d}_l.txt'.format(idx)) 
        paths['gt_right'] = os.path.join(self.gt_root_dir, 'gt_{:010d}_r.txt'.format(idx)) 
        paths['test_left'] = os.path.join(self.test_root_dir, '{:010d}_l.txt'.format(idx)) 
        paths['test_right'] = os.path.join(self.test_root_dir, '{:010d}_r.txt'.format(idx)) 
        return paths

    def loadBoundaryCoeffs(self, fn):
        coeffs = []
        f =  open(fn, 'r')
        data = list(f)
        f.close()
        if data[0] == 'null\n':
            return []
        else:
            lst = data[0].split('\n')[0].split(' ')
            return [float(i) for i in lst]

    def thirdOrderFunc(self, coeffs, x):
        print(coeffs[0]*pow(x, 3) + coeffs[1]*pow(x, 2) + coeffs[2]*pow(x, 1) + coeffs[3]) 
        return coeffs[0]*pow(x, 3) + coeffs[1]*pow(x, 2) + coeffs[2]*pow(x, 1) + coeffs[3] 

    def evaluate(self, test_path, gt_path):
        test_coeffs = self.loadBoundaryCoeffs(test_path)
        gt_coeffs = self.loadBoundaryCoeffs(gt_path)

        if len(test_coeffs) == 0:
            print("No detection from proposed method.")
            return 0.
        thres = 0.5 
        cnt, total_sample = 0, 1500
        for i in range(1500):
            x = 0.01 * i
            if abs(self.thirdOrderFunc(gt_coeffs, x) - self.thirdOrderFunc(test_coeffs, x)) < thres:
                cnt += 1
        accuracy = float(cnt) / total_sample
        print('Accuracy : {:10f}'.format(accuracy))
        return accuracy

    def runEvaluation(self):
        for idx in range(index_range[0], index_range[1]+1):
            print('---------- Frame {:5}: ----------'.format(idx))
            paths = self.getFileNames(idx) 
            
            accuracy_l = self.evaluate(paths['test_left'], paths['gt_left'])
            accuracy_r = self.evaluate(paths['test_right'], paths['gt_right'])

    def printResult(self):
        pass

gt_root_dir = '/home/rtml/Lidar_curb_detection/source/lidar_based/pointcloud_annotation/pcl_viewer/evaluation_result'
test_root_dir = '/home/rtml/Lidar_curb_detection/source/lidar_based/curb_detection_cpp/evaluation_result'
data_path = {'test_data': test_root_dir, 'ground_truth_data': gt_root_dir}

if __name__ == '__main__':
    index_range = [1750, 1755]
    evaluator = Eval(data_path, index_range)
    evaluator.runEvaluation()

#!/usr/bin/env python

# Evaluate the accuracy of curb detection
import os
import glob

class Eval:
    def __init__(self, data_paths, index_range):
        self.frames = 0
        # self.totalFrames = index_range[1]-index_range[0]+1
        self.totalFrames = 0
        self.gt_root_dir = data_paths['ground_truth_data']
        self.test_root_dir = data_paths['test_data']
        self.maxRange = 20.0
        self.step = 0.2
        self.totalSamples = int(self.maxRange / self.step)
        self.leftTable = self.initTable()
        self.rightTable = self.initTable()

    def initTable(self):
        return {'tp':[0]*self.totalSamples, 'tn':[0]*self.totalSamples, 'fp':[0]*self.totalSamples, 'fn':[0]*self.totalSamples, 'diff':[0.0]*self.totalSamples, 'Precision':[0.0]*self.totalSamples, 'Recall':[0.0]*self.totalSamples}

    def getFileNames(self, idx):
        paths = {}
        paths['gt_left'] = os.path.join(self.gt_root_dir, 'gt_{:010d}_l.txt'.format(idx)) 
        paths['gt_right'] = os.path.join(self.gt_root_dir, 'gt_{:010d}_r.txt'.format(idx)) 
        paths['test_left'] = os.path.join(self.test_root_dir, '{:010d}_l.txt'.format(idx)) 
        paths['test_right'] = os.path.join(self.test_root_dir, '{:010d}_r.txt'.format(idx)) 
        return paths

    def loadBoundaryCoeffs(self, fn):
        coeffs = []
        try:
            f =  open(fn, 'r')
        except:
            return {'coeffs':[]}
        data = list(f)
        f.close()
        if data[0] == 'null\n':
            return {'coeffs':[]}
        else:
            lst = data[0].split('\n')[0].split(' ')
            pointList = []
            for i in range(1, len(data)):
                point_str = data[i].split('\n')[0].split(' ')
                point_y = float(point_str[1])
                pointList.append(point_y)
            return {'coeffs':[float(i) for i in lst], 'min_y':min(pointList), 'max_y':max(pointList)}

    def thirdOrderFunc(self, coeffs, x):
        return coeffs[0]*pow(x, 3) + coeffs[1]*pow(x, 2) + coeffs[2]*pow(x, 1) + coeffs[3] 

    # def evaluate(self, test_path, gt_path):
    #     test_data= self.loadBoundaryCoeffs(test_path)
    #     gt_data = self.loadBoundaryCoeffs(gt_path)
    #     test_coeffs = test_data['coeffs']
    #     gt_coeffs = gt_data['coeffs']
        
    #     if len(gt_coeffs) == 0:
    #         print("No ground truth.")
    #         return 0.
    #     if len(test_coeffs) == 0:
    #         print("No detection from proposed method.")
    #         return 0.
    #     thres = 0.3
    #     cnt = 0
    #     for i in range(self.totalSamples):
    #         x = self.step * i
    #         diff_abs = abs(self.thirdOrderFunc(gt_coeffs, x) - self.thirdOrderFunc(test_coeffs, x))
    #         if diff_abs < thres:
    #             cnt += 1
    #             self.rightTable['tp'][i] += 1
    #         self.rightTable['diff'][i] += diff_abs
    #     accuracy = float(cnt) / self.totalSamples
    #     print('Accuracy : {:10f}'.format(accuracy))
    #     return accuracy

    def evaluate(self, test_path, gt_path):
        test_data= self.loadBoundaryCoeffs(test_path)
        gt_data = self.loadBoundaryCoeffs(gt_path)
        test_coeffs = test_data['coeffs']
        gt_coeffs = gt_data['coeffs']
        if len(gt_coeffs) == 0:
            print("No ground truth.")
            return 0.
        if len(test_coeffs) == 0:
            print("No detection from proposed method.")
            return 0.
        
        self.totalFrames += 1 
        min_y_test, max_y_test = test_data['min_y'], test_data['max_y']
        min_y_gt, max_y_gt = gt_data['min_y'], gt_data['max_y']
        thres = 0.3
        cnt = 0
        
        print(min_y_test, max_y_test)
        print(min_y_gt, max_y_gt)
        for i in range(self.totalSamples):
            x = self.step * i
            # No detection result
            if x < min_y_test or x > max_y_test:
                # True negative
                if x < min_y_gt or x > max_y_gt:
                    self.rightTable['tn'][i] += 1
                # False negative
                else:
                    self.rightTable['fn'][i] += 1
            else:
                # False positive
                if x < min_y_gt or x > max_y_gt:
                    self.rightTable['fp'][i] += 1
                # True positive
                else:
                    diff_abs = abs(self.thirdOrderFunc(gt_coeffs, x) - self.thirdOrderFunc(test_coeffs, x))
                    # True positive
                    if diff_abs < thres:
                        cnt += 1
                        self.rightTable['tp'][i] += 1
                    # False positive     
                    else:
                        self.rightTable['fp'][i] += 1
                    self.rightTable['diff'][i] += diff_abs
        
        accuracy = float(cnt) / self.totalSamples
        print('Accuracy : {:10f}'.format(accuracy))
        return accuracy

    def runEvaluation(self):
        for idx in range(index_range[0], index_range[1]+1):
            print('---------- Frame {:2}: ----------'.format(idx))
            paths = self.getFileNames(idx) 
            
            # accuracy_l = self.evaluate(paths['test_left'], paths['gt_left'])
            accuracy_r = self.evaluate(paths['test_right'], paths['gt_right'])
        for i in range(self.totalSamples):
            try:
                self.rightTable['Precision'][i] = float(self.rightTable['tp'][i]) / (self.rightTable['tp'][i] + self.rightTable['fp'][i])
            except:
                self.rightTable['Precision'][i] = -1.0
            try:
                self.rightTable['Recall'][i] = float(self.rightTable['tp'][i]) / (self.rightTable['tp'][i] + self.rightTable['fn'][i]) 
            except:
                self.rightTable['Recall'][i] = -1.0 
        self.printResult()

    def printResult(self):
        print('=========================')
        f = open("result.txt", "w")
        for i in range(self.totalSamples):
            print('{:05.2f} m: {{ TP:{:3d}/{}, FP: {:3d}/{}, TN:{:3d}/{}, FN: {:3d}/{}, Error Dist: {:10.5f}, Precision: {:6.3f}, Recall: {:6.3f} }}'.format(i*self.step, self.rightTable['tp'][i], self.totalFrames, self.rightTable['fp'][i], self.totalFrames, self.rightTable['tn'][i], self.totalFrames, self.rightTable['fn'][i], self.totalFrames, self.rightTable['diff'][i], self.rightTable['Precision'][i], self.rightTable['Recall'][i]))
            f.write('{:05.2f} m: {{ TP:{:3d}/{}, FP: {:3d}/{}, TN:{:3d}/{}, FN: {:3d}/{}, Error Dist: {:10.5f}, Precision: {:6.3f}, Recall: {:6.3f} }}\n'.format(i*self.step, self.rightTable['tp'][i], self.totalFrames, self.rightTable['fp'][i], self.totalFrames, self.rightTable['tn'][i], self.totalFrames, self.rightTable['fn'][i], self.totalFrames, self.rightTable['diff'][i], self.rightTable['Precision'][i], self.rightTable['Recall'][i]))
        f.close()
    
# Data paths
gt_root_dir = '/home/rtml/Lidar_curb_detection/source/evaluation/gt_generator/evaluation_result'
test_root_dir = '/home/rtml/Lidar_curb_detection/source/vscan_based/detection_result'
data_path = {'test_data': test_root_dir, 'ground_truth_data': gt_root_dir}

if __name__ == '__main__':
    index_range = [601, 650]
    evaluator = Eval(data_path, index_range)
    evaluator.runEvaluation()

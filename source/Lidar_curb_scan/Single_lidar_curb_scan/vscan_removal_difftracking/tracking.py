import numpy as np
import os
import pickle
import math 
from scipy import interpolate
from scipy.interpolate import interp1d
import sys 

class Filter():
    def __init__(self):
        #Transition matrix
        self.F_t =  np.array([ [1 ,0] , [0,1 ]])

        #Initial State cov
        self.P_t = np.identity(2)*0.2

        #Process cov
        self.Q_t = np.identity(2)

        #Control matrix
        self.B_t = np.array([ [1 ,0] , [0,1]])

        #Measurment Matrix
        self.H_t = np.array([ [1, 0], [ 0, 1]])

        #Measurment cov
        self.R_t= np.identity(2)*0.1

        #intial state
        self.X_hat_t = np.array( [[0],[0]] )

        # Control vector
        self.U_t =  np.array([ [0] , [1]])

    def prediction(self, X_hat_t_1  , P_t_1 , F_t  , B_t  ,   U_t, Q_t ):
        
        # Estimate state
        X_hat_t = F_t.dot( X_hat_t_1 )+( B_t.dot(U_t).reshape(B_t.shape[0],-1) )
        
        # Estimate covariance
        P_t = np.diag( np.diag( F_t.dot(P_t_1).dot(F_t.transpose()) ) ) + Q_t
        
        return X_hat_t , P_t
    

    def update(self, X_hat_t,P_t,Z_t,R_t,H_t):
        
        K_prime = P_t.dot( H_t.transpose() ).dot( np.linalg.inv ( H_t.dot(P_t).dot(H_t.transpose()) +R_t ) )  
        X_t = X_hat_t + K_prime.dot( Z_t - H_t.dot( X_hat_t ))
        P_t = P_t - K_prime.dot( H_t ).dot( P_t )
        
        return X_t,P_t

    def display_state(self, old, est, measure, new):
        print(" old state:", old)
        print(" Estimate:", est)
        print(" Measure:", measure)
        print(" New:", new)
        
    def main(self, points):
        old = self.X_hat_t
        
#         self.Q_t *= abs(np.random.normal(0,1))**2
        
        # predict
        X_hat_t , P_hat_t = self.prediction(self.X_hat_t , self.P_t , self.F_t , self.B_t ,self.U_t, self.Q_t)
        
        # Measure
        Z_t = np.array(points).reshape(-1,1)
        
        # Correct
        X_t , P_t = self.update(X_hat_t , P_hat_t ,Z_t ,self.R_t ,self.H_t )
        
        # Update
        self.X_hat_t = X_t
        P_hat_t=P_t
        
        self.Q_t *= abs(np.random.normal(0,points[0]))
        
        self.display_state(old, X_hat_t, points.tolist(),  X_t )
        
        return self.X_hat_t

def cubic(k, a, b, c, d):
    return a * pow(k,3) + b * pow(k,2) + c * pow(k,1) + d

def filter_process(points, line_coeffs, frame_num, line_option ):
    print("FRAME NUM IS ", frame_num)
    if frame_num <= 10:
            f = [Filter() for i in range(5)]
            prev_coeff = line_coeffs
            prev_sampled_pts = np.zeros(5)
    else:
        file = open('filters_{}.pkl'.format(line_option), 'rb')
        instance_dict = pickle.load(file)
        
        file.close()
        prev_coeff = instance_dict["prev_coeff_{}".format(line_option)]
        prev_sampled_pts = instance_dict["prev_sampled_pts_{}".format(line_option)]
        print("PREV COEFF IS ", prev_coeff)
        f = [instance_dict['0'], instance_dict['1'], instance_dict['2'], instance_dict['3'], instance_dict['4']]
        with open('filters_init_pts_{}.pkl'.format(line_option), 'wb') as output:
            pickle.dump(np.array(points), output, pickle.HIGHEST_PROTOCOL)

    # filter 
    # print("Starting to sample points ...")
    if len(points) > 6*3:
        try:
            points = np.array(points).reshape(-1, 3)  
            xy = points[:, :2]
            min_point = min(map(tuple,xy))
            max_point = max(map(tuple,xy))
            minix = min_point[0]; maxix = max_point[0]
            miniy = min_point[1]; maxiy = max_point[1]
            x = np.arange(minix, maxix, (maxix-minix)/20)
            y = np.arange(miniy, maxiy, (maxiy-miniy)/20)
            p_2 = np.polyfit(y, x, 3)
            
#             x = np.arange(minix, maxix, (maxix-minix)/5)
            x = np.arange(miniy, maxiy, (maxiy-miniy)/5)
            p2 = np.poly1d(p_2)
            p2_points  = cubic(x, *p2)
            x_regular, y_regular = x, p2_points

        except Exception as e:
            print(p2)
            print(e)
            print("FOUND THE ABOVE ERRROR IN SAMPLING, TAKING PREVIOUS POINTS")

        print("Starting filter instances ...")

        new_x, new_y = [], []
        for j,each_pt in enumerate(zip(x_regular, y_regular)):

            each = np.array(each_pt)
            leng = round( np.sqrt(each[0]**2+each[1]**2), 3)
            angle = round( np.arctan(-each[1]/each[0]), 3) 
            leng_dot = np.ediff1d(leng, to_begin=0)
           
            old_state_list = [leng, angle]
            try:
                old_state = np.array( old_state_list, dtype = np.float )
                print(" again ")
                print(old_state_list)
                old_state = old_state.reshape(-1,1)
                print(old_state.tolist())
#                 if frame_num == 3:
#                     exit()
            except Exception as e:
                print(e)
                exit()
            
            new_state = f[j].main( old_state )
         
            leng = new_state[0]
            angle = new_state[1]
            
            try:
                leng = new_state[3]
            except:
                pass
            
            endy_new =  leng* -math.sin(angle)
            endx_new = leng * math.cos(angle)
            new_x.append(endx_new)
            new_y.append(endy_new)
            new_sampled_pts = np.array([new_x, new_y])

        with open('filters_{}.pkl'.format(line_option), 'wb') as output:
            instance_dict = {}
            for i, each in enumerate(f):
                instance_dict[str(i)] = each
            instance_dict["prev_coeff_{}".format(line_option)] = line_coeffs
            instance_dict["prev_sampled_pts_{}".format(line_option)] = new_sampled_pts
            pickle.dump(instance_dict, output, pickle.HIGHEST_PROTOCOL)

        with open('filters_sampled_pts_{}.pkl'.format(line_option), 'wb') as output:
            pickle.dump(new_sampled_pts, output, pickle.HIGHEST_PROTOCOL)
    else:
        points = new_sampled_pts
    
    # Diff tracker
    try:
        prev_coeff = np.array(prev_coeff)
        line_coeffs = np.array(line_coeffs)
        print("______________DIFF COEFS_____________", prev_coeff - line_coeffs)
        print("______________DIFF POINTS_____________", np.sum(prev_sampled_pts - new_sampled_pts))

    except Exception as e:
        print(e)
        print("Error in diff tracker")
#         print(prev_coeff)
        
    
    
    return new_sampled_pts, np.sum(prev_coeff - line_coeffs), np.abs(np.sum(prev_sampled_pts - new_sampled_pts))
    
def kalman_filter_chk(lpoints, rpoints, l1, r1, prev_l_coeffs, prev_r_coeffs, frame_num):
    c = 0
    print(" Getting pyobject ")
    if(lpoints and rpoints):
        print("len of lpoints : ", len(lpoints))
        c+=1
        # print("Frame num ", frame_num)
        # Start Filter process when present
        # new_l_points = filter_process(lpoints, l1, frame_num, 'l' )
        new_r_points, coeff_diff, point_diff = filter_process(rpoints, r1, frame_num, 'r' )
        
        try:
            print(new_r_points[0].reshape(-1))
            x_pts = new_r_points[0].reshape(-1).tolist()
            y_pts = new_r_points[1].reshape(-1).tolist()
#             x_pts.reverse()
#             y_pts.reverse()
            z = np.polyfit(x_pts, y_pts, 3)
            z = z.tolist()
            z.reverse()
        except Exception as e:
            print(e)
            print("ERROR")
            exit()
            
        print("PREVIOUS Points : ", rpoints)
        print("NEW points : ", new_r_points)
        print("PREVIOUS COEFFS : ", r1)
        print("NEW COEFFS : ", z)
        saved = {} 
        saved["prev_pts"] = rpoints
        saved["new_pts"] = new_r_points
        saved["prev_coeff"] = r1
        saved["new_coeff"] = z
        np.save("state_update_chars", saved)
        
        print("_______________ both update _____________________")
        
        if point_diff < 5:
            r1 = z
        else:
            print("_______________ Using previous _____________________")
            r1 = prev_r_coeffs
#         if z[0] < 5:
#             r1 = z
        
    else:
        if(l1):
            print "lline present"
            c+=1
            if(len(l1) == 1):
                l1 = [0,0,0,0]
                c-=1
        else:
            print("something wrong")

        if(r1):
            print "rline present"
            c+=1
            if(len(r1) == 1):
                r1 = [0,0,0, 0]
                c-=1   
            else:
                new_r_points, coeff_diff, point_diff = filter_process(rpoints, r1, frame_num, 'r' )
                print("HERE3")
                try:
                    print(new_r_points[0].reshape(-1))
                    x_pts = new_r_points[0].reshape(-1).tolist()
                    y_pts = new_r_points[1].reshape(-1).tolist()
#                     x_pts.reverse()
#                     y_pts.reverse()
                    z = np.polyfit(x_pts, y_pts, 3)
                    z = z.tolist()
                    z.reverse()
                except Exception as e:
                    print(e)
                    print("ERROR")
                    exit()
                print("PREVIOUS COEFFS : ", r1)
                print("NEW COEFFS : ", z)
                print("_______________ R update _____________________")
                
                
        else:
            print("something wrong")

    # # if(c == 3):
    # if(1):
    #     cwd = os.getcwd()
    #     path = os.path.join(cwd, "point_results")
    #     data={'id': frame_num, 'l':l1, 'r': r1, 'lp': lpoints, 'rp': rpoints }
    #     try:
    #         np.save(path+"/"+str(int(frame_num)), data)
    #         print("Saved    results to file")
    #     except Exception as e:
    #         print(e)
    #         print("could not save numpy ")
    print(l1)
    print(r1)
    l1.extend(r1)
    print(l1)
    print("extended len is ", len(l1))
    
    print("Showing res ")
    print("res_curve[0] ", l1[0])
    return l1
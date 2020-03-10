import numpy as np
import os
import pickle
import math 
from scipy import interpolate
from scipy.interpolate import interp1d

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
        
        print("old state:", old, " Estimate:", X_hat_t, " Measure:", points, " New:", X_t)

        return self.X_hat_t

def filter_process(points, line_coeffs, frame_num, line_option ):
    print("FRAME NUM IS ", frame_num)
    if frame_num <= 5:
            f = [Filter() for i in range(5)]
    else:
        file = open('filters_{}.pkl'.format(line_option), 'rb')
        instance_dict = pickle.load(file)
        file.close()
        f = [instance_dict['0'], instance_dict['1'], instance_dict['2'], instance_dict['3'], instance_dict['4']]
        with open('filters_init_pts_{}.pkl'.format(line_option), 'wb') as output:
            pickle.dump(np.array(points), output, pickle.HIGHEST_PROTOCOL)
        print(f)

    # filter 
    print("Starting to sample points ...")
    
    if len(points) > 6*3:
        try:
            points = np.array(points).reshape(-1, 3)
            x = points[:, 0].tolist()
            y = points[:, 1].tolist()
#             a,b,c,d = line_coeffs
            mini = min(x); maxi = max(x);

#             print("sorted list ...")
#             x_vals= sorted(np.linspace(mini, maxi, len(points)), reverse=True)
#             line_coeffs = [a * math.pow(k,3) + b * math.pow(k,2) + c * math.pow(k,1) + d for k in x_vals]
            print("Interpolating ...")
            tck,u = interpolate.splprep(np.array(map(tuple,points[:,:2])).T.tolist())
            unew = np.arange(mini, maxi, 0.01)
            out = interpolate.splev(np.linspace(0, 1, 1000), tck)

            x1,y1 = out

            print("Cum sum ...")
            distance = np.cumsum(np.sqrt( np.ediff1d(x1, to_begin=0)**2 + np.ediff1d(y1, to_begin=0)**2 ))
            distance = distance/distance[-1]

            fx, fy = interp1d( distance, x1 ), interp1d( distance, y1 )
            alpha = np.linspace(0, 1, 5)
            x_regular, y_regular = fx(alpha), fy(alpha)

        except Exception as e:
            print(e)
            print("FOUND THE ABOVE ERRROR IN SAMPLING, TAKING PREVIOUS POINTS")

        print("Starting filter instances ...")

        new_x, new_y = [], []
        for j,each_pt in enumerate(zip(x_regular, y_regular)):

            each = np.array(each_pt)
            leng = np.sqrt(each[0]**2+each[1]**2)
            angle = np.arctan(-each[1]/each[0])

            new_state = f[j].main(np.array([[leng], [angle]]))
            print("Old state: ", np.array([[leng], [angle]]).tolist(), " New state: ", new_state.tolist())

            leng = new_state[0]
            angle = new_state[1]
            endy_new =  leng* -math.sin(angle)
            endx_new = leng * math.cos(angle)
            new_x.append(endx_new)
            new_y.append(endy_new)
            new_sampled_pts = np.array([new_x, new_y])

        with open('filters_{}.pkl'.format(line_option), 'wb') as output:
            instance_dict = {}
            for i, each in enumerate(f):
                instance_dict[str(i)] = each
            pickle.dump(instance_dict, output, pickle.HIGHEST_PROTOCOL)

        with open('filters_sampled_pts_{}.pkl'.format(line_option), 'wb') as output:
            pickle.dump(new_sampled_pts, output, pickle.HIGHEST_PROTOCOL)
    else:
        points = new_sampled_pts
    
    return new_sampled_pts
    
def kalman_filter_chk(lpoints, rpoints, l1, r1, frame_num):
    
    print("len of l : ", len(l1))
    print("len of r : ", len(r1))

    c = 0
    print(" Getting pyobject ")
    if(lpoints and rpoints):
        print("len of lpoints : ", len(lpoints))
        print "points present"
        c+=1
        # print("Frame num ", frame_num)
        # Start Filter process when present
        # new_l_points = filter_process(lpoints, l1, frame_num, 'l' )
#         new_r_points = filter_process(rpoints, r1, frame_num, 'r' )
#         print("HERE3")
#         try:
# #             print(new_r_points[0].reshape(-1))
#             z = np.polyfit(new_r_points[0].reshape(-1), new_r_points[1].reshape(-1), 3)
#         except Exception as e:
#             print(e)
#             print("ERROR")
#         print("PREVIOUS COEFFS : ", r1)
#         print("NEW COEFFS : ", z)
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
    return l1
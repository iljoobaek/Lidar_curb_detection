import numpy as np
import os
def kalman_filter(state,inputs, z,u):
    """Function to compute the Extended Kalman Filter Gain"""
    # ......Prediction Step....................
    # Predicting the next state priori from previous state
    print(" Getting pyobject ")
#     X_prior=np.matmul(self.F,self.X)+u*self.Bd_state
#     # Predicting the error covariance
#     P_prior=np.matmul(np.matmul(self.F,self.P),np.transpose(self.F))+self.W
#     # ......Correction Step....................
#     # Computing the Kalman Gain
#     temp=np.matmul(np.matmul(self.H,P_prior),np.transpose(self.H))
#     K=np.matmul(np.matmul(P_prior,np.transpose(self.H)),np.linalg.inv(temp))
#     # Update the estimate with measurement
#     self.X=X_prior+np.matmul(K,z-np.matmul(self.H,X_prior))
#     # Update the error covariance
#     self.P=np.matmul(np.eye(4)-np.matmul(K,self.H),P_prior)
    return np.array([[1,2,3],[12,34,5]]).tolist()



# class Tracking():
#     def kalman_filter_chk(self,val):
#         if val:
#             print("value is : ", val)
#         print(" Getting pyobject ")
#         return 0


class Filter():
    def __init__(self):
        #Transition matrix
        self.F_t = np.array([ [1 ,0, delta_t ,0] , [0,1,0, delta_t ] , [0,0,1,0] , [0,0,0,1] ])

        #Initial State cov
        self.P_t = np.identity(4)*0.2

        #Process cov
        self.Q_t = np.identity(4)

        #Control matrix
        self.B_t = np.array( [ [0] , [0], [0] , [0] ])

        #Control vector
        self.U_t = acceleration

        #Measurment Matrix
        self.H_t = np.array([ [1, 0, 0, 0], [ 0, 1, 0, 0]])

        #Measurment cov
        self.R_t= np.identity(2)*5

        #intial state
        self.X_hat_t = np.array( [[0],[0],[0],[0]] )

    def prediction(self, X_hat_t_1  , P_t_1 , F_t  , B_t  ,   U_t, Q_t ):
        X_hat_t=F_t.dot(X_hat_t_1)+(B_t.dot(U_t).reshape(B_t.shape[0],-1) )
        P_t=np.diag(np.diag(F_t.dot(P_t_1).dot(F_t.transpose())))+Q_t
        return X_hat_t , P_t
    

    def update(self, X_hat_t,P_t,Z_t,R_t,H_t):
        
        K_prime = P_t.dot( H_t.transpose() ).dot( np.linalg.inv ( H_t.dot(P_t).dot(H_t.transpose()) +R_t ) )  
        print("K:\n",K_prime)
        
        X_t = X_hat_t + K_prime.dot( Z_t - H_t.dot( X_hat_t ))
        P_t = P_t - K_prime.dot( H_t ).dot( P_t )
        
        return X_t,P_t

    def main(self, points, l1, r1, frame_num):
        self.points = points
        X_hat_t , P_hat_t = self.prediction(self.X_hat_t , P_t , F_t , B_t , U_t, Q_t)
        print("Prediction:")
        print("X_hat_t:\n",X_hat_t,"\nP_t:\n",P_t)
        
        Z_t = self.measurmens.transpose()
        Z_t = Z_t.reshape( Z_t.shape[0], -1) # [[ ],[ ],[ ],[ ]]
        
        print(Z_t.shape)
        
        X_t , P_t = self.update(X_hat_t , P_hat_t ,Z_t ,R_t ,H_t )
        print("Update:")
        print("X_t:\n",X_t,"\nP_t:\n",P_t)
        self.X_hat_t = X_t
        P_hat_t=P_t


def kalman_filter_chk(lpoints, rpoints, l1, r1, frame_num):
    print("len of l : ", len(l1))
    print("len of r : ", len(r1))
    c = 0
    print " Getting pyobject "
    if(lpoints and rpoints):
        print "points present"
        c+=1
        
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

    # if(c == 3):
    if(1):
        cwd = os.getcwd()
        path = os.path.join(cwd, "point_results")
        data={'id': frame_num, 'l':l1, 'r': r1, 'lp': lpoints, 'rp': rpoints }
        try:
            np.save(path+"/"+str(int(frame_num)), data)
            print("Saved    results to file")
        except Exception as e:
            print(e)
            print("could not save numpy ")
    print(l1)
    print(r1)
    l1.extend(r1)
    print(l1)
    print("extended len is ", len(l1))
    return l1
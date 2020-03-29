'''
utils provides helper classes and methods.
'''

from enum import Enum
import numpy as np
import numpy.matlib

from math import sin, cos, sqrt, atan2

def state_vector_to_scalars(state_vector):
    '''
    Returns the elements from the state_vector as a tuple of scalars.
    '''
    return (state_vector[0][0,0],state_vector[1][0,0],state_vector[2][0,0],state_vector[3][0,0])

def cart_2_polar(state_vector):
    '''
    Transforms the state vector into the polar space.
    '''

    px,py,vx,vy = state_vector_to_scalars(state_vector)
    ro      = sqrt(px**2 + py**2)

    phi     = atan2(py,px)
    ro_dot  = (px*vx + py*vy)/ro

    return np.matrix([ro, phi, ro_dot]).T

def polar_2_cart(ro, phi, ro_dot):
    '''
    ro: range
    phi: bearing
    ro_dot: range rate

    Takes the polar coord based radar reading and convert to cart coord x,y, vx, vy
    return (x,y, vx, vy)
    '''
    return (cos(phi) * ro, sin(phi) * ro, ro_dot * cos(phi) , ro_dot * sin(phi))

def passing_rmse(rmse, metric):
    print("Metric: ", metric)
    for index, threshold in enumerate(metric):
        if rmse[0,index] > threshold:
            print("RMSE FAILED metric @ index ", index)
            return False

    print("RMSE PASSED metric")
    return True

def calculate_rmse(estimations, ground_truth):
    '''
    Root Mean Squared Error.
    '''
    if len(estimations) != len(ground_truth) or len(estimations) == 0:
        raise ValueError('calculate_rmse () - Error - estimations and ground_truth must match in length.')

    rmse = np.matrix([0.,0.,0.,0.]).T

    for est, gt in zip(estimations, ground_truth):
        rmse += np.square(est - gt)

    rmse /= len(estimations)
    return np.sqrt(rmse)


class SensorType(Enum):
    '''
    Enum types for the sensors. Future sensors would be added here.
    '''
    LIDAR = 'L'
    RADAR = 'R'

class MeasurementPacket:
    '''
    Defines a measurement datapoint for multiple sensor types.
    '''
    def __init__(self, packet):
        self.sensor_type = SensorType.LIDAR if packet[0] == 'L' else SensorType.RADAR

        if self.sensor_type == SensorType.LIDAR:
            self.setup_lidar(packet)
        elif self.sensor_type == SensorType.RADAR:
            self.setup_radar(packet)

    def setup_radar(self, packet):
        self.rho_measured       = packet[1]
        self.phi_measured       = packet[2]
        self.rhodot_measured    = packet[3]
        self.timestamp          = packet[4]
        self.x_groundtruth      = packet[5]
        self.y_groundtruth      = packet[6]
        self.vx_groundtruth     = packet[7]
        self.vy_groundtruth     = packet[8]

    def setup_lidar(self, packet):
        self.x_measured         = packet[1]
        self.y_measured         = packet[2]
        self.timestamp          = packet[3]
        self.x_groundtruth      = packet[4]
        self.y_groundtruth      = packet[5]
        self.vx_groundtruth     = packet[6]
        self.vy_groundtruth     = packet[7]

    @property
    def z(self):
        '''
        Returns a vectorized version of the measurement for EKF typically called z.
        '''
        if self.sensor_type == SensorType.LIDAR:
            return np.matrix([[self.x_measured,self.y_measured]]).T
        elif self.sensor_type == SensorType.RADAR:
            return np.matrix([[self.rho_measured,self.phi_measured,self.rhodot_measured]]).T

    @property
    def ground_truth(self):
        return np.matrix([[self.x_groundtruth,
                         self.y_groundtruth,
                         self.vx_groundtruth,
                         self.vy_groundtruth]]).T

    def __str__(self):
        if self.sensor_type == SensorType.LIDAR:
            return "LIDAR (timestamp: {:>8}) \n MEASUREMENT [{:>4} || {:>4}] \n GROUND TRUTH [{:>4} || {:>4} || {:>4} || {:>4}]".format(
                    self.timestamp,

                    self.x_measured,
                    self.y_measured ,

                    self.x_groundtruth ,
                    self.y_groundtruth,
                    self.vx_groundtruth,
                    self.vy_groundtruth)

        elif self.sensor_type == SensorType.RADAR:
            return "RADAR (timestamp: {:>8}) \n MEASUREMENT [{:>4} || {:>4} <> {:>4}] \n GROUND TRUTH [{:>4} || {:>4} || {:>4} || {:>4}]".format(
                    self.timestamp    ,

                    self.rho_measured,
                    self.phi_measured ,
                    self.rhodot_measured,

                    self.x_groundtruth,
                    self.y_groundtruth ,
                    self.vx_groundtruth,
                    self.vy_groundtruth )

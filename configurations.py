import mock
import numpy as np
def defaultConfig():
    config = mock.Mock()

    config.dt = 0.01

    #Shape of vehicle characteristics
    config.m = 3.2
    config.Iyy = 1/12*config.m*0.8**2
    config.S = 0.25
    config.c = 0.13

    #Aero parameters
    config.Cl_0 = 0.5
    config.Cl_alpha = 0.1*180/np.pi
    config.Cd_0 = 0.1
    config.K = 0.05
    config.Cm_0 = 0.5
    config.Cm_alpha = -0.14*180/np.pi
    config.Cm_alpha_dot = -0.008*180/np.pi
    config.Cm_delta_e = 0.2

    #Random values
    config.rhoSTD = 0
    config.Restimation = np.diag([0.25, (0.25/180*np.pi)**2])


    return config
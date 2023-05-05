import numpy as np
class Drone():
    g = 9.81
    rhoNom = 1.225

    #Init function that coppies over the config file into the class variables.
    def __init__(self, config):
        self.dt = config.dt 
        self.m = config.m
        self.Iyy = config.Iyy
        self.s = config.S
        self.Cl_0 = config.Cl_0
        self.Cl_alpha = config.Cl_alpha
        self.K = config.K
        self.Cm_0 = config.Cm_0
        self.Cm_alpha = config.cm_alpha_dot
        self.Cm_alpha_dot = config.Cm_alpha_dot
        self.Cm_delta_e = config.Cm_delta_e
        self.rhoSTD = config.rhoSTD
        self.Restimation = config.Restimation

        self.limits = np.array([
            config.x_thres,
            config.z_thresHigh,
            config.z_thresLow,
            config.v_thres,
            config.theta_thres,
            config.theta_dot_thres,
            config.gamma_thres
        ])

        self.state = None
        self.prev_action = None
        self.steps_beyond_terminated = None
        self.time = 0.
        self.objects = None
import numpy as np
import scipy as sci

class Drone():
    g = 9.81
    rhoNom = 1.225

    #Init function that coppies over the config file into the class variables.
    def __init__(self, config):
        self.dt = config.dt 
        self.m = config.m
        self.Iyy = config.Iyy
        self.S = config.S
        self.Cl_0 = config.Cl_0
        self.Cl_alpha = config.Cl_alpha
        self.K = config.K
        self.Cm_0 = config.Cm_0
        self.Cd_0 = config.Cd_0
        self.Cm_alpha = config.Cm_alpha
        self.Cm_alpha_dot = config.Cm_alpha_dot
        self.Cm_delta_e = config.Cm_delta_e
        self.rhoSTD = config.rhoSTD
        self.Restimation = config.Restimation

        self.x_thres = config.x_thres
        self.z_thresHigh = config.z_thresHigh
        self.z_thresLow = config.z_thresLow
        self.v_thres = config.v_thres
        self.theta_thres = config.theta_thres
        self.theta_dot_thres = config.theta_dot_thres
        self.gamma_thres = config.gamma_thres


        self.state = None
        self.prev_action = None
        self.steps_beyond_terminated = None
        self.time = 0.
        self.objects = None
 

    #Calculates nominal veloticty and flight path angle given elevator and thrust command
    def vGammaFromElevator(self, T, delta_e):
        alphaNom = -(self.Cm_0+self.Cm_delta_e*delta_e)/self.Cm_alpha
        Clnom = self.Cl_0 + self.Cl_alpha*alphaNom
        Cdnom = self.Cd_0 + self.K*Clnom**2
        gammaNom = Cdnom/Clnom*(1-T/self.m/self.g*np.sin(alphaNom)) + T*np.cos(alphaNom)/self.m/self.g
        vNom = np.sqrt(2*(-T*np.sin(alphaNom)+self.m*self.g*np.cos(gammaNom))/(self.S*self.rhoNom))
        return vNom, gammaNom

    #Given a thrust and pitch, finds the necesary elevator command 
    def solveForElevator(self, T, thetaDes):
        #Calculates pitch angle from 
        def thetaFromElevator(self, T,thetaDes, delta_e):
            alphaNom = -(self.Cm_0+self.Cm_delta_e*delta_e)/self.Cm_alpha
            Clnom = self.Cl_0 + self.Cl_alpha*alphaNom
            Cdnom = self.Cd_0 + self.K*Clnom**2
            return thetaDes- (-Cdnom/Clnom + T/(self.m*self.g)*(1-Cdnom/Clnom*alphaNom))-alphaNom
        func = lambda x: thetaFromElevator(T, thetaDes,x)
        sol = sci.optimize.root_scalar(func, method='secant', x0 = 0, x1 = 1)
        return sol.root
    
    #Calculates the elevator command given a desired angle of attack
    def elevatorFromAlpha(self, alpha):
        return -(self.Cm_alpha*alpha+self.Cm_0)/self.Cm_delta_e 

    # Calculates a "coherent" (steady state stable) command from thrust and gamma command
    # Returns airspeed, theta, thetadot, and flight path angle (gamma)
    def coherentCommand(self, T, gamma):
        airspeedGuess = 15
        thetaGuess = 0
        thetaDot = 0
        verticalForces = lambda X: ((self.Cl_0 + self.Cl_alpha*(X[0]- gamma))*1/2*(X[1]**2)*self.rhoNom*self.S - self.m*self.g*np.cos(gamma) + T*np.sin(X[0]-gamma), \
               ((self.Cd_0+self.K*(self.Cl_0 + self.Cl_alpha*(X[0]- gamma))**2)*1/2*(X[1]**2)*self.rhoNom*self.S -T*np.cos(X[0]-gamma)-self.m*self.g*np.sin(gamma)))

        sol = sci.optimize.root(verticalForces, x0 = [thetaGuess, airspeedGuess])
        X =sol.x      
        return [sol.x[1], sol.x[0], thetaDot, gamma]
    


import numpy as np
import scipy as sci
import Plant
import control as ctrl
import gymnasium as gym

class Drone():
    g = 9.81
    rhoNom = 1.225

    #Init function that coppies over the config file into the class variables.
    def __init__(self, config, configPlant):
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
        self.Pestimation = config.Pestimation
        self.Qestimation = config.Qestimation

        self.state = None
        self.prev_action = None
        self.steps_beyond_terminated = None
        self.time = 0.
        self.objects = None

        self.plant = Plant.Plant(configPlant)
        self.numObstacles = config.numObstacles

        self.Qcontrol = config.Qcontrol
        self.Rcontrol = config.Rcontrol

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
    
    # Call this function to update the estimated state of the system
    def updateEKF(self, controlIn, observation):
        v, theta, theta_dot, gamma = self.stateEstimate
        alpha = theta-gamma
        q = 0.5*self.rhoNom*v**2

        #Calculate alphaDot
        Cl = self.Cl_0 + self.Cl_alpha*alpha
        L = q*self.S*Cl
        gamma_dot = (L - self.m*self.g*np.cos(gamma) +controlIn[0]*np.sin(alpha)) / (self.m*v)
        alpha_dot = theta_dot-gamma_dot

        #Other aero
        Cd = self.Cd_0 + self.K*Cl**2
        Cm = self.Cm_0 + self.Cm_alpha*alpha + self.Cm_alpha_dot*alpha_dot + self.Cm_delta_e*controlIn[1]
        D = q*self.S*Cd
        M = q*self.S*Cm

        #derivatives
        v_dot = (-D - self.m*self.g*np.sin(gamma) + controlIn[0]*np.cos(alpha)) / self.m
        #theta_dot = theta_dot
        theta_ddot = M / self.Iyy
        gamma_dot = (L - self.m*self.g*np.cos(gamma) +controlIn[0]*np.sin(alpha)) / (self.m*v)

        AFull, BFull = self.calculateCTSABMatrix([v, theta, theta_dot, gamma], controlIn)
        F = AFull*self.dt + np.eye(4) 

        #Observability 
        H = np.array([[1.0,0.0,0.0,0.0],[0.0,1.0,0.0,0.0]])

        #predict
        v += v_dot*self.dt
        theta += theta_dot*self.dt 
        theta_dot += theta_ddot * self.dt
        gamma += gamma_dot*self.dt
        xEst = np.array([[v], [theta], [theta_dot], [gamma]])
        
        self.Pestimation = (F)@self.Pestimation @ np.transpose(F) + self.Qestimation
        #update
        y = np.array([[v],[theta]]) - np.array([observation[0],observation[1]])
        S = H @ self.Pestimation @ H.T + self.Restimation
        K = self.Pestimation @ np.transpose(H) @ sci.linalg.inv(S)
        xEst -= K @ y
        self.Pestimation = (np.eye(4) - K @ H) @ self.Pestimation
        self.stateEstimate = xEst.flatten()

    # Returns the state and control Jacobians
    def calculateCTSABMatrix(self, stateIn, controlIn):
        (thrust, delta_e) = controlIn
        v, theta, theta_dot, gamma = stateIn
        q = 0.5*self.rhoNom*v**2
        alpha = theta-gamma

        #Calculate alphaDot
        Cl = self.Cl_0 + self.Cl_alpha*alpha
        L = q*self.S*Cl
        gamma_dot = (L - self.m*self.g*np.cos(gamma) +thrust*np.sin(alpha)) / (self.m*v)
        alpha_dot = theta_dot-gamma_dot

        #Other aero
        Cd = self.Cd_0 + self.K*Cl**2
        Cm = self.Cm_0 + self.Cm_alpha*alpha + self.Cm_alpha_dot*alpha_dot + self.Cm_delta_e*delta_e
        D = q*self.S*Cd
        M = q*self.S*Cm

        #Partial derivatives of aeroforces
        pDpv = self.rhoNom*v*self.S*Cd
        pDptheta = 2*self.K*q*self.S*(self.Cl_alpha * self.Cl_0 + self.Cl_alpha**2*alpha)
        pDpthetaDot = 0
        pDpgamma = -2*self.K*q*self.S*(self.Cl_alpha* self.Cl_0+self.Cl_alpha**2*alpha)
        pMpv = self.rhoNom*v*self.S*Cm
        pMptheta = q*self.S*self.Cm_alpha
        pMpthetaDot = 0
        pMpgamma = -q*self.S*self.Cm_alpha
        pLpv = self.rhoNom*v*self.S*Cl
        pLptheta = q*self.Cl_alpha*self.S
        pLpthetaDot = 0
        pLpgamma = -q*self.Cl_alpha*self.S

        #out of order entries
        pf4ptheta = (pLptheta + thrust*np.cos(alpha))/(self.m*v)
        pf4pgamma = (pLpgamma/self.m +self.g*np.sin(gamma)-thrust/self.m*np.cos(alpha))/v

        #Partial derivatives of nonlinear map with respect to tstate
        pf1pv = - 1/self.m*pDpv
        pf1ptheta = (-pDptheta-thrust*np.sin(alpha))/self.m
        pf1pthetaDot = 0
        pf1pgamma = (-pDpgamma/self.m - self.g*np.cos(gamma) + thrust/self.m*np.sin(alpha))
        pf2pv = 0
        pf2ptheta = 0
        pf2pthetaDot = 1
        pf2pgamma = 0
        pf3pv = pMpv/self.Iyy
        pf3ptheta = pMptheta/self.Iyy
        pf3pthetaDot = 0
        pf3pgamma = pMpgamma/self.Iyy
        pf4pv = pLpv/(self.m*v)
        pf4pthetaDot = 0

        AFull = np.array([[pf1pv, pf1ptheta, pf1pthetaDot, pf1pgamma],[pf2pv, pf2ptheta, pf2pthetaDot, pf2pgamma],
                      [pf3pv, pf3ptheta, pf3pthetaDot, pf3pgamma],[pf4pv, pf4ptheta, pf4pthetaDot, pf4pgamma]])
        BFull = np.array([[0], [0], [q*self.Cm_delta_e*self.S],[0]])
        return AFull, BFull

    def calculateGains(self, initState):
        x, y, v, theta, theta_dot, gamma = initState
        AFull, BFull =  self.calculateCTSABMatrix(initState[2:], (4, self.elevatorFromAlpha(theta - gamma)))
        self.KFull = ctrl.lqr(AFull, BFull, self.Qcontrol, self.Rcontrol)[0]

    # Resets the state of the system and creates obstacles
    def reset(self, initState):
        self.plant.reset(initState)
        self.stateEstimate = initState[2:]
        self.droneTime = 0
        self.calculateGains(initState)
        self.delta_e_actual = 0

        # Setting up obstacles
        x_pos = np.linspace(initState[0]+40,initState[0]+240,self.numObstacles)
        x_pos = np.hstack((x_pos,x_pos))
        y_pos = np.random.uniform(low=initState[1]-8,high=initState[1]-2,size=self.numObstacles)
        y_pos = np.hstack((y_pos,y_pos+8))
        sizes = np.random.uniform(low=1,high=1,size=self.numObstacles*2)
        self.objects = np.stack((x_pos,y_pos,sizes))

        return self.state
    
    def step(self, actionDrone):
        # Control inputs
        thrust, delta_eFF, stateCommand  = actionDrone

        error = (np.array([self.stateEstimate]).T-np.array([stateCommand]).T)
        controlFull = self.KFull @ error
        delta_e = delta_eFF-controlFull[0][0]
        actionPlant = (thrust, delta_e)
        self.delta_e_actual = delta_e

        observation, reward, terminated = self.plant.step(actionPlant)
        self.updateEKF(actionPlant, observation)
        self.droneTime += self.dt

        if not terminated:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            reward = 0.0
        else:
            if self.steps_beyond_terminated == 0:
                gym.logger.warn(
                    "You are calling 'step()' even though this "
                    "environment has already returned terminated = True. You "
                    "should always call 'reset()' once you receive 'terminated = "
                    "True' -- any further steps are undefined behavior."
                )
            self.steps_beyond_terminated += 1
            reward = 0.0

        #self.update_plot(self.axs[0],state,self.lidar,self.objects)
        #rays = self.update_plot(self.axs[1],state,self.lidar,self.objects,beams=True)
        #self.grid = self.update_grid(self.axs[2],self.lidar,rays)

        return self.stateEstimate, reward, terminated#, rays.astype(int)


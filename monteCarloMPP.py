import Drone
from configurations import defaultConfig 
import numpy as np
from ConvexMotionPlanning import TrajGenMPP 
from plottingFunctions import state_plots_command, open_loop_plots
samples = 20
success = [0, 0, 0, 0, 0]
obstacles = [0, 5, 10, 15, 20]
for successIter in range(len(success)):
    for i in range(samples):
        # Drone instantiation
        config = defaultConfig()
        config.numObstacles = obstacles[successIter]
        drone = Drone.Drone(config, config)

        # Parameters
        thrustCommandVal = 4.3
        refStates = drone.coherentCommand(thrustCommandVal,0/180*np.pi)
        controlRef = (thrustCommandVal, drone.elevatorFromAlpha(refStates[1]-refStates[3]))
        iterations = 2000
        iter = 0
        pathDist = 100
        stateRef = np.hstack(([0,50], refStates))
        drone.reset(stateRef)

        # Data
        traj = np.zeros((7,iterations))
        elvActual = np.zeros(iterations)
        thrustCommand = np.zeros(iterations)+thrustCommandVal
        stateCommand = drone.coherentCommand(thrustCommand[0],0/180*np.pi)
        elivCommandRef = drone.elevatorFromAlpha(stateCommand[1]-stateCommand[3])
        elivCommand = np.zeros(iterations) + elivCommandRef
        xCommand = np.zeros(iterations)
        yCommand = np.zeros(iterations)
        refCommand = np.zeros((iterations, 4))
        refCommand[0] = refStates

        # Main loop
        iter = 0
        terminated = False
        while iter < iterations and (not terminated):
            action = [thrustCommand[iter], elivCommand[iter], refCommand[iter, :]] 
            observation, grid, reward, terminated = drone.step(action)
            traj[:,iter] = np.hstack((drone.plant.state,drone.plant.time))
            elvActual[iter] = drone.delta_e_actual
            if(iter%pathDist == 0):
                try:
                    thrustCommand, elivCommand, xCommand, yCommand, refCommand = TrajGenMPP(drone, grid, thrustCommand, elivCommand, xCommand, yCommand, refStates, refCommand, thrustCommandVal,elivCommandRef, iterations, iter)
                except:
                    terminated = True
            iter += 1
        if (not terminated):
            success[successIter] = success[successIter] + 1
    print(successIter)
print(success)
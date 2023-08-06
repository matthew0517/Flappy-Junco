import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import Drone
from configurations import defaultConfig 
import numpy as np
from ConvexMotionPlanning import TrajGenMPP 
from plottingFunctions import state_plots_command, open_loop_plots


# Drone instantiation
config = defaultConfig()
config.numObstacles = 20
drone = Drone.Drone(config, config)

# Parameters
thrustCommandVal = 4.3
refStates = drone.coherentCommand(thrustCommandVal,0/180*np.pi)
controlRef = (thrustCommandVal, drone.elevatorFromAlpha(refStates[1]-refStates[3]))
iterations = 2500
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
for iter in range(0, iterations):
    action = [thrustCommand[iter], elivCommand[iter], refCommand[iter, :]] 
    observation, grid, reward, terminated = drone.step(action)
    traj[:,iter] = np.hstack((drone.plant.state,drone.plant.time))
    elvActual[iter] = drone.delta_e_actual
    if(iter%pathDist == 0):
        thrustCommand, elivCommand, xCommand, yCommand, refCommand = TrajGenMPP(drone, grid, thrustCommand, elivCommand, xCommand, yCommand, refStates, refCommand, thrustCommandVal,elivCommandRef, iterations, iter)
    if(terminated):
        break

vRef = refCommand[:,0]
thetaRef = refCommand[:,1]
gammaRef = refCommand[:,3]
x = traj[0,:]
y = traj[1,:]
v = traj[2,:]
theta = traj[3,:]
thetaDot = traj[4,:]
gamma = traj[5,:]
time = traj[6,:]

# Create a 3x2 grid for subplots
fig = plt.figure(figsize=(10, 8.25))
gs = GridSpec(3, 2, height_ratios=[3, 2, 2], width_ratios=[1, 1])

# Big subplot in the top center
big_subplot = fig.add_subplot(gs[0, 0:2])

# Subplots in the first column
small_subplot_1 = fig.add_subplot(gs[1, 0])
small_subplot_2 = fig.add_subplot(gs[2, 0])

# Subplots in the second column
small_subplot_3 = fig.add_subplot(gs[1, 1])
small_subplot_4 = fig.add_subplot(gs[2, 1])

# Plot your data in each subplot
# For example:
big_subplot.scatter(drone.objects[0,:],drone.objects[1,:], color = "red",  label="obstacles" )
big_subplot.set_ylim((35, 65))
big_subplot.plot(xCommand, yCommand, label="reference command")
big_subplot.plot(x, y,  label="true trajectory")
big_subplot.set_ylabel("Height (m)")
big_subplot.set_xlabel("Downrange distance (m)")
big_subplot.grid(True)
big_subplot.legend(loc="upper right") 

small_subplot_1.plot(time,gammaRef*180/np.pi)
small_subplot_1.plot(time,gamma*180/np.pi)
small_subplot_1.set_ylabel("Flight Path Angle (deg)")
small_subplot_1.set_xlabel("Time (s)")
small_subplot_1.grid(True)

small_subplot_2.plot(time,thetaRef*180/np.pi)
small_subplot_2.plot(time,theta*180/np.pi)
small_subplot_2.set_ylabel("Pitch (deg)")
small_subplot_2.set_xlabel("Time (s)")
small_subplot_2.grid(True)


small_subplot_3.plot(time,vRef)
small_subplot_3.plot(time,v)
small_subplot_3.set_ylabel("Airspeed (m/s)")
small_subplot_3.set_xlabel("Time (s)")
small_subplot_3.grid(True)

small_subplot_4.plot(time,thrustCommand)
small_subplot_4.set_ylabel("Thrust Command (N)")
small_subplot_4.set_xlabel("Time (s)")
small_subplot_4.grid(True)
# Adjust layout and add legend
#plt.tight_layout()
#plt.legend()

# Show the plot
plt.tight_layout()
plt.savefig('figure5.png', bbox_inches='tight')
plt.show()

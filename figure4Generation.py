import Drone
from configurations import defaultConfig 
import numpy as np
from plottingFunctions import open_loop_plots, plot_rrt_lines_flipped
import importlib
import ConvexMotionPlanning
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.axes import Axes
import RRTAR
import copy

## Part I perception
config = defaultConfig()
config.numObstacles = 40
drone = Drone.Drone(config, config)

thrustCommand = 4.3
fpaCommand = 0/180*np.pi
dynRef = drone.coherentCommand(thrustCommand,fpaCommand)
elivatorRef = drone.elevatorFromAlpha(dynRef[1]-dynRef[3])
stateRef = np.hstack(([0,50], dynRef))
action = (thrustCommand, elivatorRef, dynRef)
drone.reset(stateRef)

observation, grid, reward, terminated = drone.step(action)


# random data
fig, ax = plt.subplots(3, 1, figsize=(5,7.5),sharex=True)

# define the colors
cmap = mpl.colors.ListedColormap(['w', 'k'])

# create a normalize object the describes the limits of
# each color
bounds = [0., 0.5, 1.]
norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

# plot it
x = drone.ogOrigin[0]
z = drone.ogOrigin[1]
theta = drone.plant.state[3]
beam = drone.lidar_range
angle = drone.lidar_angle
res = drone.lidar_res
ax[0].plot([x,x+beam*np.cos(theta+angle/2)],[z,z+beam*np.sin(theta+angle/2)],'k--')
ax[0].plot([x,x+beam*np.cos(theta-angle/2)],[z,z+beam*np.sin(theta-angle/2)],'k--')
a = np.linspace(-angle/2,angle/2,int(res/3))
xs = x+beam*np.cos(theta+a)
zs = z+beam*np.sin(theta+a)
ax[0].plot(xs,zs,'k.',markersize=1)
ax[0].scatter(25, 37, color='red', s=100)


ax[0].imshow(grid, interpolation='none', cmap=cmap, norm=norm)
#ax.set_xlabel("Downrange distance (m)")
#ax.set_ylabel("Vertical distance (m)")
ax[0].set_xlim(0,100)
ax[0].grid(True)
plt.gca().invert_yaxis()
ax[0].set_ylim(20,60)


##PART II


midpoint = drone.ogOrigin[1]
offset = 10
lookAhead = 54
orignX = drone.ogOrigin[0]
xstart = np.array([orignX, offset]) 
xgoal = [[orignX+lookAhead, offset], [orignX+lookAhead, offset/2],  [orignX+lookAhead, offset/4*2],  [orignX+lookAhead, offset*3/2]]
n = 10
r_rewire = 5
searchGrid = grid[midpoint-offset:midpoint+offset, 0:85].T.astype(int)
vectorsSave = []
limitIter = 100
li = 0

while(len(vectorsSave) < 50 and li < limitIter):
    vectors = RRTAR.runRRT(xstart[0], xstart[1], 150, 2.5, 5, searchGrid, xgoal)
    li = li+1
    for i in range(1, len(vectors)):
        vectorsSave.append(vectors[i])
print(len(vectorsSave))
#tree_plot_2d_vectors(vectors[0], grid[midpoint-offset:midpoint+offset, 0:85].T.astype(int))
for i in range(0, len(vectorsSave)):
    vectors = vectorsSave[i]
    for i in range(len(vectors)):
        x = vectors[i][0]
        y = vectors[i][1]+27

        if i > 0:
            target_x = vectors[i-1][0]
            target_y = vectors[i-1][1]+27

            ax[1].plot([x, target_x], [y, target_y], 'b-')
        ax[1].scatter(x, y, color='royalblue', s=10)

    #ax.set_xlabel('Downrange distance (m)')
    ax[1].grid(True)

og=grid[0:midpoint+offset, 0:85].T.astype(int)
norm = Normalize(vmin=0, vmax=1)
ax[1].imshow(og.T, cmap=cmap, norm=norm, origin="lower", interpolation=None)
ax[1].set_xlim(0,100)
plt.gca().invert_yaxis()
ax[1].set_ylim(20,60)
ax[1].grid(True)
ax[1].scatter(25, 37, color='red', s=100)

## Part III
import math
import importlib
importlib.reload(ConvexMotionPlanning)
thrustCommandVal = 4.3
refStates = drone.coherentCommand(thrustCommandVal,0/180*np.pi)
controlRef = (thrustCommandVal, drone.elevatorFromAlpha(refStates[1]-refStates[3]))

iterations = 2000
iter = 0
thrustCommand = np.zeros(iterations)+thrustCommandVal
stateCommand = drone.coherentCommand(thrustCommand[0],0/180*np.pi)
elivCommandRef = drone.elevatorFromAlpha(stateCommand[1]-stateCommand[3])
elivCommand = np.zeros(iterations) + elivCommandRef
xCommand = np.zeros(iterations)
yCommand = np.zeros(iterations)
xCommandfin = np.zeros(iterations)
yCommandfin = np.zeros(iterations)
thrustCommandFin = np.zeros(iterations)
elivCommandFin= np.zeros(iterations)

refCommand = np.zeros((iterations, 4))
timeMulti = 25
xstart = np.concatenate(([0.,0.], np.array(drone.stateEstimate) - np.array(refStates) ))
xgoal = np.concatenate(([0.,50 - drone.plant.state[1]], [0.,0.,0.,0.]))
tEnd = 450

minCost = 10000000

ax[2].imshow(grid[0:midpoint+offset, 0:85].T.astype(int).T, cmap=cmap, norm=norm, origin="lower", interpolation=None)
ax[2].set_xlim(0,100)
ax[2].set_ylim(20,60)
ax[2].scatter(25, 37, color='red', s=100)

ax[2].grid(True)

for vi in range(len(vectorsSave)):
    Acts, Bcts = drone.calculateCTSABMatrix(refStates, controlRef)
    Aopt = np.eye(Acts.shape[0]+2)
    #A = Aopt + env.calculateANumerical(stateRefFull, controlRef, env.rhoNom, step=10**-5)*env.dt
    Aopt[0, 2] = drone.dt*timeMulti
    Aopt[1, 5] = refStates[0]*drone.dt*timeMulti
    Aopt[2:,2:] = Aopt[2:,2:]+Acts*drone.dt*timeMulti
    Bopt = np.zeros([6,2])
    Bopt[2:,0:1] = Bcts*drone.dt*timeMulti
    alphaEst = refStates[1] - refStates[3]
    Bopt[2, 1] = 1/drone.m*np.cos(alphaEst)*drone.dt*timeMulti
    Bopt[5, 1] = 1/drone.m*np.sin(alphaEst)/refStates[0]*drone.dt*timeMulti

    path_pts = np.zeros((len(vectorsSave[vi]) - 1, 2, 2))
    for i in range(len(vectorsSave[vi]) - 1):
        path_pts[i] = np.array([[vectorsSave[vi][i][1], vectorsSave[vi][i][0]],
                        [vectorsSave[vi][i+1][1], vectorsSave[vi][i+1][0]]])
    referencePoints, referenceVels = ConvexMotionPlanning.calculateReferencePoints(math.floor(tEnd/timeMulti)+1, path_pts)
    referencePointsDyn, referenceVelsDyn = ConvexMotionPlanning.calculateReferencePoints(math.floor(tEnd/timeMulti)+1, path_pts)
    for i in range(len(referencePointsDyn)):
        referencePointsDyn[i][0] = referencePointsDyn[i][0]-refStates[0]*np.sin(refStates[3])*i*drone.dt*timeMulti-offset
        referencePointsDyn[i][1] = referencePointsDyn[i][1]-refStates[0]*np.cos(refStates[3])*i*drone.dt*timeMulti-orignX

    xsol, usol, cost, infeasbility = ConvexMotionPlanning.localTrajOpt(Aopt, Bopt, math.floor(tEnd/timeMulti), grid[midpoint-offset:, 0:80], referencePoints, referencePointsDyn, xstart, xgoal)
    if (infeasbility < 0.01):
        for i in range(np.min((tEnd-timeMulti-1, iterations-iter))):
            iDrop = math.floor(i/timeMulti)
            iNext = iDrop+1
            ifrac1 = (timeMulti - i%timeMulti)/timeMulti
            ifrac2 = (i%timeMulti)/timeMulti

            thrustCommand[iter+i] = ifrac1*usol[iDrop][1] + ifrac2*usol[iNext][1]  + thrustCommandVal 
            elivCommand[iter + i] = ifrac1*usol[iDrop][0] + ifrac2*usol[iNext][0] + elivCommandRef
            xCommand[iter+i] = ifrac1*xsol[iDrop][0] + ifrac2*xsol[iNext][0] +refStates[0]*np.cos(0)*i*drone.dt+drone.plant.state[0]
            yCommand[iter+i] = ifrac1*xsol[iDrop][1] + ifrac2*xsol[iNext][1] + refStates[0]*np.sin(0)*i*drone.dt+drone.plant.state[1]
            refCommand[iter+i, :] = ifrac1*xsol[iDrop][2:6].T + ifrac2*xsol[iNext][2:6].T + np.array(refStates)
        ax[2].plot(xCommand[0:tEnd-timeMulti-1]+25, yCommand[0:tEnd-timeMulti-1]-13, color = 'lightblue')
        if(cost < minCost):
            minCost = cost
            thrustCommandFin = thrustCommand+0
            elivCommandFin = elivCommand+0
            xCommandfin = xCommand+0
            yCommandfin = yCommand+0

    else:
        for i in range(np.min((tEnd-timeMulti-1, iterations-iter))):
            iDrop = math.floor(i/timeMulti)
            iNext = iDrop+1
            ifrac1 = (timeMulti - i%timeMulti)/timeMulti
            ifrac2 = (i%timeMulti)/timeMulti

            thrustCommand[iter+i] = ifrac1*usol[iDrop][1] + ifrac2*usol[iNext][1]  + thrustCommandVal 
            elivCommand[iter + i] = ifrac1*usol[iDrop][0] + ifrac2*usol[iNext][0] + elivCommandRef
            xCommand[iter+i] = ifrac1*xsol[iDrop][0] + ifrac2*xsol[iNext][0] +refStates[0]*np.cos(0)*i*drone.dt+drone.plant.state[0]
            yCommand[iter+i] = ifrac1*xsol[iDrop][1] + ifrac2*xsol[iNext][1] + refStates[0]*np.sin(0)*i*drone.dt+drone.plant.state[1]
            refCommand[iter+i, :] = ifrac1*xsol[iDrop][2:6].T + ifrac2*xsol[iNext][2:6].T + np.array(refStates)
        ax[2].plot(xCommand[0:tEnd-timeMulti-1]+25, yCommand[0:tEnd-timeMulti-1]-13, color = 'grey')
ax[2].plot(xCommandfin[0:tEnd-timeMulti-1]+25, yCommandfin[0:tEnd-timeMulti-1]-13, color = 'blue')



fig.text(0.04, 0.5, 'Vertical distance (m)', va='center', rotation='vertical')
ax[2].set_xlabel('Downrange distance (m)')
plt.savefig('figure4.png', bbox_inches='tight')
plt.show()
import numpy as np
import math
from scipy.spatial import Delaunay
from cvxopt import matrix, solvers
from rrtplanner import RRTStar

# Calculates the length of a path given as a series of points
def pathlens(path_points):
    dist = np.zeros(path_points.shape[0])
    for i in range(path_points.shape[0]):
        dist[i] = math.dist(path_points[i, 0,:], path_points[i,1,:])
    return dist

# Calculates an occupancy grid given convex constraints
def calculateOccupancyGrid(constraints):
    yv, xv = np.meshgrid(range(0, 200), range(0, 200))
    og = np.zeros_like(xv)

    for constraint in constraints:
        hull = Delaunay(constraint)
        ogInner = np.zeros_like(xv)
        len, wid = xv.shape
        for i in range(len):
            for j in range(wid):
                ogInner[i, j] = hull.find_simplex([xv[i,j],yv[i,j]])
        og[ogInner != -1] = 1
    return og

# Resamples a path into a series of points that can be optimized.  These points are evenly spaced.
def calculateReferencePoints(timeEnd, path_points):
    pathLengths = pathlens(path_points)
    vel = sum(pathLengths)/timeEnd
    distanceOfPoints = np.linspace(start = 0, stop = sum(pathLengths), num = timeEnd)
    referencePoints = []
    referencePoints.append(path_points[0, 0, :])
    referenceVels = []
    pathLengths[-1] = pathLengths[-1] + 10**-3
    index = 0
    for i in range(1, distanceOfPoints.size):
        while(distanceOfPoints[i] > pathLengths[index]):
            distanceOfPoints = distanceOfPoints - pathLengths[index]
            index = index + 1
        multiplier = distanceOfPoints[i]/pathLengths[index]
        referencePoints.append(path_points[index, 0, :] *(1-multiplier) + path_points[index, 1, :]*multiplier)
        referenceVels.append(referencePoints[i]- referencePoints[i-1])
    referenceVels.append((referencePoints[0]- referencePoints[1]) * 0)

    return referencePoints, referenceVels

# Performs a search of the occupency grid along a single direction
def singleRay(origin, direction, og, limit = 10):
    direction[direction == 0] = 10**-8
    cord = np.array(origin)
    traverse = 0
    while traverse < limit:
        distance = np.floor(cord+1) - cord
        distance[direction < 0] = np.ceil(cord[direction < 0] -1) - cord[direction < 0]
        ratio = np.abs(distance / direction)
        len = np.amin(ratio)
        cord = len*direction + cord
        if ((np.greater((np.array([cord.astype(int)])+1),  og.shape)).any()) or (og[tuple(cord.astype(int))]):
            return np.floor(cord)
        traverse += len
    return np.floor(cord)

# Performs a search of the occupency grid in multiple directions from a single point.
def searchFromCord(cord, directions, og, limit = 10):
    sol = []
    for direct in directions:
        sol.append(singleRay(cord, direct, og, limit))
    return np.array(sol)

# Returns the directions used in the searchFromCord function
def generateDirections(numDirects = 8):
    angles = np.linspace(0, 2*np.pi*(numDirects-1)/(numDirects), numDirects)
    sol = []
    for ang in angles:
        sol.append([np.cos(ang), np.sin(ang)])
    return np.array(sol)


# The main function for this library.  Search as occupency grid around the given reference points to create a set of linearly constrained open points.  
# These points are optimized to minimize a control effort and comply with the system dynamics.
def localTrajOpt(A, B, tEnd, og, referencePoints, referencePointsDyn, xstart, xgoal):
    dimX = len(A)
    dimU = B.shape[1]


    #Inequality constraints
    numDirections = 8
    directs = generateDirections(numDirections)
    Gbar = np.zeros(((numDirections)*math.floor((tEnd)/10+1)+tEnd+1, (dimX+dimU)*tEnd+dimX))
    hbar = np.zeros(((numDirections)*math.floor((tEnd)/10+1)+tEnd+1, 1))
    Refs0 = list(zip(*referencePoints))[0]
    Refs1 = list(zip(*referencePoints))[1]
    RefsDyn0 = list(zip(*referencePointsDyn))[0]
    RefsDyn1 = list(zip(*referencePointsDyn))[1]

    for i in range(0,tEnd+1,10):
        cords = searchFromCord([Refs0[i],Refs1[i]], directs, og, limit = 20)
        imod = math.floor(i/10)
        hbar[imod*numDirections:(imod+1)*numDirections, 0] = np.sum(directs*(cords-np.array([Refs0[i],Refs1[i]])+np.array([RefsDyn0[i],RefsDyn1[i]])), axis=1)-1*i/tEnd
        Gbar[imod*numDirections:(imod+1)*numDirections, i*(dimX+dimU):i*(dimX+dimU)+2] = np.flip(directs, axis = 1)

    # Adding positive control constraints
    print(tEnd +(math.floor(tEnd/10)+1)*numDirections )
    for i in range(tEnd):
        hbar[(math.floor(tEnd/10)+1)*numDirections+i, 0] = 4.4
        Gbar[(math.floor(tEnd/10)+1)*numDirections+i, i*(dimX+dimU)+ dimX+1] = -1
    hbar = hbar

    IA = np.eye(dimX)
    Abar = np.zeros((dimX*(tEnd+1),(dimX+dimU)*tEnd+dimX))
    Bbar = np.zeros((dimX*(tEnd+1),1))

    #Equality constraints
    for i in range(tEnd):
        leftInd = i*(dimX+dimU)
        righInd =  (i+1)*(dimX+dimU)+dimX
        upInd = i*dimX
        botInd = (i+1)*dimX
        Abar[upInd:botInd, leftInd:righInd] = np.hstack([A, B, -IA])
        Bbar[upInd:upInd+len(referencePointsDyn[i])] = 0
    leftInd = 0
    righInd =  dimX
    upInd = tEnd*dimX
    botInd = (tEnd+1)*dimX
    Abar[upInd:botInd, leftInd:righInd] = IA
    Bbar[upInd:botInd,0] = xstart

    #Cost function
    finalStateWeight = 100
    Qbar = np.zeros(((dimX+dimU)*(tEnd)+dimX,(dimX+dimU)*(tEnd)+dimX))
    IU = np.eye(dimU)
    psiDotWeight = 1000*0
    for i in range(tEnd):
        psiDotIndex = i*(dimX+dimU)+5
        Qbar[psiDotIndex, psiDotIndex] = psiDotWeight

        leftInd = i*(dimX+dimU)+dimX
        righInd =  (i)*(dimX+dimU)+dimU+dimX
        upInd = i*(dimX+dimU)+dimX
        botInd = (i)*(dimX+dimU) +dimU+dimX
        Qbar[upInd:botInd, leftInd:righInd] = IU
    leftInd = tEnd*(dimX+dimU)
    righInd =  (tEnd)*(dimX+dimU)+dimX
    upInd =tEnd*(dimX+dimU)
    botInd =  (tEnd)*(dimX+dimU)+dimX
    Qbar[upInd:botInd, leftInd:righInd] = np.eye(dimX)*finalStateWeight

    Pbar = np.zeros(((dimX+dimU)*(tEnd)+dimX,1))
    Pbar[upInd:botInd, 0] = -xgoal*finalStateWeight
    
    #Run CVX
    Qop = matrix(Qbar)
    pop = matrix(Pbar)
    Gop = matrix(Gbar)
    hop = matrix(hbar)
    Aop = matrix(Abar)
    bop = matrix(Bbar)
    sol = solvers.qp(Qop, pop, Gop, hop, Aop, bop)
    z = sol['x']

    #Output solution
    xsol = []
    usol = []
    for i in range(tEnd):
        xsol.append(z[i*(dimX+dimU):i*(dimX+dimU)+dimX])
        usol.append(z[i*(dimX+dimU)+dimX:i*(dimX+dimU)+dimX+dimU])
    xsol.append(z[(tEnd)*(dimX+dimU)])
    return xsol, usol

# Trajectory Generation Demo rolled into a single function to make the demo cleaner
def TrajGen(drone, grid, thrustCommand, elivCommand, xCommand, yCommand, refStates, refCommand, thrustCommandVal,elivCommandRef, iterations, iter):
    controlRef = (thrustCommandVal, elivCommandRef)
    midpoint = drone.ogOrigin[1]
    offset = 10
    lookAhead = 54
    orignX = drone.ogOrigin[0]
    xstart = np.array([offset, orignX]) 
    xgoal = np.array([offset, orignX+lookAhead])
    n = 3000
    r_rewire = 5
    rrts = RRTStar(grid[midpoint-offset:midpoint+3, 0:80], n, r_rewire, pbar = False) 
    T, gv = rrts.plan(xstart, xgoal)
    path = rrts.route2gv(T, gv)
    path_pts = rrts.vertices_as_ndarray(T, path)
    tEnd = 450

    referencePoints, referenceVels = calculateReferencePoints(tEnd+1, path_pts)
    referencePointsDyn, referenceVelsDyn = calculateReferencePoints(tEnd+1, path_pts)
    for i in range(len(referencePointsDyn)):
        referencePointsDyn[i][0] = referencePointsDyn[i][0]-refStates[0]*np.sin(refStates[3])*i*drone.dt-offset
        referencePointsDyn[i][1] = referencePointsDyn[i][1]-refStates[0]*np.cos(refStates[3])*i*drone.dt-orignX
    Acts, Bcts = drone.calculateCTSABMatrix(refStates, controlRef)
    Aopt = np.eye(Acts.shape[0]+2)
    #A = Aopt + env.calculateANumerical(stateRefFull, controlRef, env.rhoNom, step=10**-5)*env.dt
    Aopt[0, 2] = drone.dt
    Aopt[1, 5] = refStates[0]*drone.dt
    Aopt[2:,2:] = Aopt[2:,2:]+Acts*drone.dt
    Bopt = np.zeros([6,2])
    Bopt[2:,0:1] = Bcts*drone.dt
    alphaEst = refStates[1] - refStates[3]
    Bopt[2, 1] = 1/drone.m*np.cos(alphaEst)*drone.dt
    Bopt[5, 1] = 1/drone.m*np.sin(alphaEst)/refStates[0]*drone.dt
    xstart = np.concatenate(([0.,0.], np.array(drone.stateEstimate) - np.array(refStates)))
    xgoal = np.concatenate(([0.,50 - drone.plant.state[1]], [0.,0.,0.,0.]))
    xsol, usol = localTrajOpt(Aopt, Bopt, tEnd, grid[midpoint-offset:, 0:80], referencePoints, referencePointsDyn, xstart, xgoal)
    for i in range(np.min((tEnd, iterations-iter))):
        thrustCommand[iter+i] = usol[i][1] + thrustCommandVal 
        elivCommand[iter + i] = usol[i][0] + elivCommandRef
        xCommand[iter+i] = xsol[i][0]+refStates[0]*np.cos(0)*i*drone.dt+drone.plant.state[0]
        yCommand[iter+i] = xsol[i][1]+refStates[0]*np.sin(0)*i*drone.dt+drone.plant.state[1]
        refCommand[iter+i, :] = xsol[i][2:6].T + np.array(refStates)
    return thrustCommand, elivCommand, xCommand, yCommand, refCommand
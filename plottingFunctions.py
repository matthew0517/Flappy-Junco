import matplotlib.pyplot as plt
import numpy as np

def state_plots(traj, elvActual):

    def plot_state(ax,x,y,xlabel,ylabel):
        lim = 2
        ax.plot(x,y)
        ax.set_ylim([min(y)-lim,max(y)+lim])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        return
    x = traj[0,:]
    z = traj[1,:]
    v = traj[2,:]
    theta = traj[3,:]
    thetaDot = traj[4,:]
    gamma = traj[5,:]
    time = traj[6,:]

    plt.close()
    fig, axs = plt.subplots(3,2, figsize=(8,8))
    plot_state(axs[0, 0],x,z,"Downrange distance (m)","Height (m)")
    plot_state(axs[0, 1],time,(theta-gamma)*180/np.pi,"Time (s)","alpha (deg)")
    plot_state(axs[1, 0],time,gamma*180/np.pi,"Time (s)","Gamma (deg)")
    plot_state(axs[1, 1],time,v,"Time (s)","Velocity (m/s)")
    plot_state(axs[2, 0],time,theta*180/np.pi,"Time (s)","Theta (deg)")
    plot_state(axs[2, 1],time,elvActual,"Time (s)","Elevator")

    fig.show()
    return

def state_plots_command(traj,control,xCommand, yCommand, elivCommand,elvActual, refCommand):
    def plot_state(ax,x,y,xlabel,ylabel):
        lim = 2
        ax.plot(x,y)
        ax.set_ylim([min(y)-lim,max(y)+lim])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        return
    vRef = refCommand[:,0]
    thetaRef = refCommand[:,1]
    gammaRef = refCommand[:,3]

    x = traj[0,:]
    z = traj[1,:]
    v = traj[2,:]
    theta = traj[3,:]
    thetaDot = traj[4,:]
    gamma = traj[5,:]
    time = traj[6,:]

    plt.close()
    fig, axs = plt.subplots(3,3, figsize=(8,8))
    plot_state(axs[0, 0],x,z,"Downrange distance (m)","Height (m)")
    plot_state(axs[0, 0],xCommand,yCommand,"Downrange distance (m)","Height (m)")
    axs[0,0].scatter(env.objects[0,:],env.objects[1,:], color = "red" )
    axs[0,0].set_ylim((90,110))
    plot_state(axs[0, 1],time,(theta-gamma)*180/np.pi,"Time (s)","alpha (deg)")
    plot_state(axs[0, 1],time,(thetaRef-gammaRef)*180/np.pi,"Time (s)","alpha (deg)")
    plot_state(axs[0, 2],time,elvActual,"Time (s)","Elevator Actual")
    plot_state(axs[0, 2],time,elivCommand,"Time (s)","Elevator Command")
    plot_state(axs[1, 0],time,gamma*180/np.pi,"Time (s)","Gamma (deg)")
    plot_state(axs[1, 0],time,gammaRef*180/np.pi,"Time (s)","Gamma (deg)")
    plot_state(axs[1, 1],time,v,"Time (s)","Velocity (m/s)")
    plot_state(axs[1, 1],time,vRef,"Time (s)","Velocity (m/s)")
    plot_state(axs[2, 0],time,theta*180/np.pi,"Time (s)","Theta (deg)")
    plot_state(axs[2, 0],time,thetaRef*180/np.pi,"Time (s)","Theta (deg)")
    plot_state(axs[2, 1],time,control,"Time (s)","Thrust Command (N)")
    fig.show()
    return

def est_plots(traj, trajEst):

    def plot_state(ax,x,y,xlabel,ylabel):
        lim = 2
        ax.plot(x,y)
        ax.set_ylim([min(y)-lim,max(y)+lim])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)
        return
    def plot_state_dashed(ax,x,y,xlabel,ylabel):
        lim = 2
        ax.plot(x,y,'--')
        ax.set_ylim([min(y)-lim,max(y)+lim])
        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        return
    
    x = traj[0,:]
    z = traj[1,:]
    v = traj[2,:]
    vEst =  trajEst[0,:]
    theta = traj[3,:]
    thetaEst = trajEst[1,:]
    thetaDot = traj[4,:]
    thetaDotEst = trajEst[2,:]
    gamma = traj[5,:]
    gammaEst = trajEst[3,:]
    time = traj[6,:]

    plt.close()
    fig, axs = plt.subplots(4,2, figsize=(8,1))
    plot_state(axs[0, 0],time,theta*180/np.pi,"Time (s)", "Theta (deg)")
    plot_state(axs[0, 1],time,thetaEst*180/np.pi,"Time (s)", "Theta Estimate (deg)")
    plot_state(axs[1, 0],time,thetaDot*180/np.pi,"Time (s)", "ThetaDot (deg/s)")
    plot_state(axs[1, 1],time,thetaDotEst*180/np.pi,"Time (s)", "ThetaDot Estimate (deg/s)")
    plot_state(axs[2, 0],time,v,"Time (s)", "Velocity (m/s)")
    plot_state(axs[2, 1],time,vEst,"Time (s)", "Velocity Estimate (m/s)")
    plot_state(axs[3, 0],time,gamma*180/np.pi,"Time (s)", "Gamma (deg)")
    plot_state(axs[3, 1],time,gammaEst*180/np.pi,"Time (s)", "Gamma Estimate (deg)")
    fig.show()
    return
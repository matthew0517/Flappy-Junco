# Flappy-Junco
A 2-d longitudinal flight sim for developing motion planning approaches.  The interface is modeled after Gymnasium however, only classical autonomy approaches are used in this project.

# Plant.py
The plant models the actual system dynamics. It recieved commands for the elevator and throttle and returns observations of the states.

The PlantDemo.ipynb is a notebook showing the step response and steady state response. 

# Drone.py
Drone contains a plant which models the physics, but it built to operate and Extended Kalman Filter and use a Linear Quadratic Regulator.  It recieves feedfoward control for throttle and elevator as well as a reference state to do feedback control around.

The DroneDemo.ipynb is a notebook showing the step response, steady state response, and filter performance. 

# Support Functions 
ConvexMotionPlanning.py is an indevelopment libarary intended to perform local trajectory optimization around a given path.  It returns feedforward controls and coherent reference states for the system to track.

PlottingFunctions.py holds the various plotting functions for the project.

import numpy as np
import time
import pickle
# import matplotlib.pyplot as plt
import random

import Franka
import RobotUtil as rt

import mujoco as mj
from mujoco import viewer


# NOTE: Please set a random seed for your random joint generator!
random.seed(13)

# Open the simulator model
xml_filepath = "../franka_emika_panda/panda_with_hand_torque.xml"

#Initialize robot object
mybot = Franka.FrankArm()

# set the initial and goal joint configurations
qInit = [-np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, 0, np.pi - np.pi/6, 0]
qGoal = [np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, 0, np.pi - np.pi/6, 0]

# Initialize some variables related to the simulation
joint_counter = 0
thresh = 0.1
FoundSolution = False
SolutionInterpolated = False

# Initializing planner variables as global for access between planner and simulator
MyPlan = []
interpolated_plan = []
plan_length = len(MyPlan)
inc = 1

# Utility function for smooth linear interpolation of RRT plan, used by the controller
def naive_interpolation(plan):

    angle_resolution = 0.01

    global interpolated_plan 
    global SolutionInterpolated
    interpolated_plan = np.empty((1,7))
    np_plan = np.array(plan)
    interpolated_plan[0] = np_plan[0]
    
    for i in range(np_plan.shape[0]-1):
        max_joint_val = np.max(np_plan[i+1] - np_plan[i])
        number_of_steps = int(np.ceil(max_joint_val/angle_resolution))
        inc = (np_plan[i+1] - np_plan[i])/number_of_steps

        for j in range(1,number_of_steps+1):
            step = np_plan[i] + j*inc
            interpolated_plan = np.append(interpolated_plan, step.reshape(1,7), axis=0)


    SolutionInterpolated = True
    print("Plan has been interpolated successfully!")


# PRM Graph query function
def PRMQuery():

    global MyPlan
    global FoundSolution

    # open road map
    f = open("myPRM.p", 'rb')
    prmVertices = pickle.load(f)
    prmEdges = pickle.load(f)
    pointsObs = pickle.load(f)
    axesObs = pickle.load(f)
    f.close

    # define start and goal
    deg_to_rad = np.pi/180. 

    #Find neighbors to initial and goal nodes
    neighInit=[]
    neighGoal=[]
    heuristic=[]
    parent=[]

    for i in range(len(prmVertices)):
        if np.linalg.norm(np.array(prmVertices[i])-np.array(qInit))<2.:
            if not mybot.DetectCollisionEdge(prmVertices[i], qInit, pointsObs, axesObs):
                neighInit.append(i)

        if np.linalg.norm(np.array(prmVertices[i])-np.array(qGoal))<2.:
            if not mybot.DetectCollisionEdge(prmVertices[i], qGoal, pointsObs, axesObs):
                neighGoal.append(i)

        heuristic.append(np.linalg.norm(np.array(prmVertices[i])-np.array(qGoal)))
        parent.append([])


    activenodes= neighInit
    bestscore=0

    while bestscore<1000 and not any([g in activenodes for g in neighGoal]):

        bestscore = 1000

        for i in range(len(activenodes)):
            for j in range(len(prmEdges[activenodes[i]])):
                if prmEdges[activenodes[i]][j] not in activenodes:
                    if heuristic[prmEdges[activenodes[i]][j]]<bestscore:
                        bestscore=heuristic[prmEdges[activenodes[i]][j]]
                        bestcandi=prmEdges[activenodes[i]][j]
                        bestparent=activenodes[i]

        if bestscore<1000:
            activenodes.append(bestcandi)
            parent[bestcandi]= bestparent


    if any([g in activenodes for g in neighGoal]):
        print("Found a path")
        FoundSolution = True
    else:
        print("Failed to find a plan")

    plan= [activenodes[-1]]
    prevstep= parent[plan[0]]
    while prevstep:
        plan.insert(0, prevstep)
        prevstep = parent[plan[0]]
    print("Plan: ", plan)

    MyPlan=[]
    MyPlan.append(qInit)
    for i in range(len(plan)):
        MyPlan.append(prmVertices[plan[i]])
    MyPlan.append(qGoal)

    naive_interpolation(MyPlan)

################################# YOU DO NOT NEED TO EDIT ANYTHING BELOW THIS ##############################
def position_control(model, data):
    global joint_counter
    global inc
    global plan_length
    global interpolated_plan

    # Instantite a handle to the desired body on the robot
    body = data.body("hand")

    # Check if plan is available, if not go to the home position
    if (FoundSolution==False or SolutionInterpolated == False):
        desired_joint_positions = np.array(qInit)
    else:
        # If a plan is available, cycle through poses
        plan_length = interpolated_plan.shape[0]

        if np.linalg.norm(interpolated_plan[joint_counter] - data.qpos[:7]) < 0.01 and joint_counter < plan_length:
            joint_counter+=inc

        desired_joint_positions = interpolated_plan[joint_counter]

        if joint_counter==plan_length-1:
            inc = -1*abs(inc)
            joint_counter-=1
        if joint_counter==0:
            inc = 1*abs(inc)
    
    # Set the desired joint velocities
    desired_joint_velocities = np.array([0,0,0,0,0,0,0])

    # Desired gain on position error (K_p)
    Kp = np.eye(7,7)*300

    # Desired gain on velocity error (K_d)
    Kd = 50

    # Set the actuator control torques
    data.ctrl[:7] = data.qfrc_bias[:7] + Kp@(desired_joint_positions-data.qpos[:7]) + Kd*(desired_joint_velocities-data.qvel[:7])

if __name__ == "__main__":
    # Load the xml file here
    model = mj.MjModel.from_xml_path(xml_filepath)
    data = mj.MjData(model)

    # Set the simulation scene to the qInit configuration
    mj.mj_resetDataKeyframe(model, data, 0)

    # Set the position controller callback
    mj.set_mjcb_control(position_control)

    # Load the PRM Graph and search it for a feasible path
    PRMQuery()

    # Launch the simulate viewer
    viewer.launch(model, data)
from random import sample, seed
from re import A
import time
import pickle
import numpy as np
import RobotUtil as rt
import Franka
import time
import mujoco as mj
from mujoco import viewer

# Seed the random object
seed(10)

# Open the simulator model from the MJCF file
xml_filepath = "../franka_emika_panda/panda_with_hand_torque.xml"

np.random.seed(0)
deg_to_rad = np.pi/180.

#Initialize robot object
mybot = Franka.FrankArm()

# Initialize some variables related to the simulation
joint_counter = 0

# Initializing planner variables as global for access between planner and simulator
plan=[]
interpolated_plan = []
plan_length = len(plan)
inc = 1

# Add obstacle descriptions into pointsObs and axesObs
pointsObs=[]
axesObs=[]

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.1,0,1.0]),[1.3,1.4,0.1])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.1,-0.65,0.475]),[1.3,0.1,0.95])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.1, 0.65,0.475]),[1.3,0.1,0.95])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[-0.5, 0, 0.475]),[0.1,1.2,0.95])
pointsObs.append(envpoints), axesObs.append(envaxes)

envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.45, 0, 0.25]),[0.5,0.4,0.5])
pointsObs.append(envpoints), axesObs.append(envaxes)

# define start and goal
deg_to_rad = np.pi/180.

# set the initial and goal joint configurations
qInit = np.array([-np.pi/2, -np.pi/2, np.pi/2, -np.pi/2, 0, np.pi - np.pi/6, 0])
qGoal = np.array([np.pi/2, -np.pi/2, -np.pi/2, -np.pi/2, 0, np.pi - np.pi/6, 0])

# Initialize some data containers for the RRT planner
rrtVertices=[] # list of vertices
rrtEdges=[] # parent of each vertex

rrtVertices.append(qInit)
rrtEdges.append(0)

thresh=0.1
FoundSolution=False
SolutionInterpolated = False

#utility function for checking whether there are collisions along a path
def check_path(q1, q2, dq_check=0.05):
    dist = vertice_distance(q1,q2)
    for i in range(int(dist/dq_check)):
        q_check = q1 + i * dq_check * (q2 - q1) / dist
        if mybot.DetectCollision(q_check, pointsObs, axesObs):
            return False
    return True

#utility function for adding a delta q to qc
def add_dq(qc, qr, dq_dist_max):
    return qc + dq_dist_max * (qr - qc) / vertice_distance(qc, qr)

#utility function for distance between two points
def vertice_distance(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2)))

# Utility function to find the index of the nearset neighbor in an array of neighbors in prevPoints
def FindNearest(prevPoints,newPoint):
    D=np.array([np.linalg.norm(np.array(point)-np.array(newPoint)) for point in prevPoints])
    return D.argmin()

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

def RRTQuery():
    global FoundSolution
    global plan
    global rrtVertices
    global rrtEdges

    goal_bias_probability = 0.1 #probability that qr is set to qGoal
    dq_dist_max = 0.3   #maximum distance between two connected nodes in q-space
    shortening_mode = 2 #0 -> no shortening, 1 -> vertex shortening, 2 -> edge shortening

    min_dist = vertice_distance(qInit, qGoal)   #used for debugging to ensure the RRT is getting closer to the goal
    # print("Distance to Goal: ", min_dist) #debug print statement

    while len(rrtVertices) < 3000 and not FoundSolution:
        # RRT algorithm to find a path to the goal configuration

        if np.random.rand(1) > goal_bias_probability:   #if not goal bias
            qr = np.array(mybot.SampleRobotConfig())    #get random robot configuration
            qr[5:] = qInit[5:]  #only randomize the first 5 joints (this reduces the dimentionality of the configuration space)
        else:   #if goal bias
            qr = qGoal  #set qr to qGoal
        
        qn_index = FindNearest(rrtVertices, qr) #find the nearest neighbor to qr in the rrtVertices

        qn = rrtVertices[qn_index]  #get qn
        qc = qn #qc starts as qn

        while vertice_distance(qc,qr) > 0:  #continue until qc is qr
            qc = add_dq(qc, qr, dq_dist_max)    #add the maximum delta q to qc that moves towards qr
            
            if check_path(qn,qc):   #if the path has no collisions
                rrtVertices.append(qc)  #add the vertex and edge to the grapth
                rrtEdges.append(qn_index)
                qn_index = len(rrtVertices) - 1
                qn = qc
                if min_dist > vertice_distance(qc, qGoal):  #check if the new node is closer to qGoal than the previous minimum
                    min_dist = vertice_distance(qc, qGoal)  #update debug variable min_dist
                    # print("Distance to Goal: ", min_dist)
            else:   #if there is a collision, break early
                break

            if vertice_distance(qc,qGoal) < dq_dist_max:    #if the new vertex is close to qGoal
                if check_path(qc,qGoal):    #if there is a feasible path to the goal from the new vertex
                    rrtVertices.append(qGoal)   #add qGoal and its edge to the graph
                    rrtEdges.append(qn_index)
                    FoundSolution = True
                    print("solution found") #report we founda solution and break
                    break

    ### if a solution was found
    if FoundSolution:
        # Extract path
        c=-1 #Assume last added vertex is at goal
        plan.insert(0, rrtVertices[c])

        while True:
            c=rrtEdges[c]
            plan.insert(0, rrtVertices[c])
            if c==0:
                break

        if shortening_mode == 0:    #no shortening
            print("no shortening")
        
        elif shortening_mode == 1:    #vertex shortening, only use vertices we already have in the graph
            print("performing vertex shortening")
            for i in range(150):    #attempt to shorten the path 150 times, vertex shortening will likely converge before 150 iterations
                q1_index = int(np.floor(np.random.rand() * len(plan))) #get two random integer indices
                q2_index = int(np.floor(np.random.rand() * len(plan)))
                while(True):    #ensure q2_index > q1_index
                    if q1_index == q2_index:
                        q2_index = int(np.floor(np.random.rand() * len(plan)))
                    elif(q1_index > q2_index):
                        temp_index = q1_index
                        q1_index = q2_index
                        q2_index = temp_index
                    else:
                        break
                if check_path(plan[q1_index],plan[q2_index]):   #if the path between vertices is feasible
                    plan = plan[:q1_index+1] + plan[q2_index:]  #shorten the path
        
        elif shortening_mode == 2:  #edge shortening, can shorten by making new vertices anywhere along the plath
            print("performing edge shortening")
            for i in range(150):    #attempt to shorten the path 150 times, edge shortening will assymptotically approach the local optimal the longer it runs
                q1_index = np.random.rand() * (len(plan) - 1) #get two random indices, not integers
                q2_index = np.random.rand() * (len(plan) - 1)
                while(True):    #ensure q2_index > q1_index, and that q1 and q2 won't be on the same line in q-space
                    if np.floor(q1_index) == np.floor(q2_index):
                        q2_index = np.random.rand() * (len(plan) - 1)
                    elif(q1_index > q2_index):
                        temp_index = q1_index
                        q1_index = q2_index
                        q2_index = temp_index
                    else:
                        break
                
                #compute new vertices for potential shortening
                q1 = plan[int(np.floor(q1_index))] * (np.ceil(q1_index) - q1_index) + plan[int(np.ceil(q1_index))] * (q1_index - np.floor(q1_index))
                q2 = plan[int(np.floor(q2_index))] * (np.ceil(q2_index) - q2_index) + plan[int(np.ceil(q2_index))] * (q2_index - np.floor(q2_index))

                if check_path(q1,q2):   #if the path between vertices is feasible
                    plan = plan[:int(np.ceil(q1_index))] + [q1,q2] + plan[int(np.ceil(q2_index)):]  #shorten the path
    
        for (i, q) in enumerate(plan):
            print("Plan step: ", i, "and joint: ", q)
    
        plan_length = len(plan)

        plan_q_length = 0
        for i in range(len(plan)-1):
            plan_q_length += vertice_distance(plan[i],plan[i+1])

        print("Plan Length in Vertices: ", plan_length) #print the number of vertices the path has
        print("Plan Length in Q: ", plan_q_length)  #print the length of the path in q-space

        naive_interpolation(plan)
        return

    else:
        print("No solution found")

################################# YOU DO NOT NEED TO EDIT ANYTHING BELOW THIS ##############################
def position_control(model, data):
    global joint_counter
    global inc
    global plan
    global plan_length
    global interpolated_plan

    # Instantite a handle to the desired body on the robot
    body = data.body("hand")

    # Check if plan is available, if not go to the home position
    if (FoundSolution==False or SolutionInterpolated==False):
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

    # Set the simulation scene to the home configuration
    mj.mj_resetDataKeyframe(model, data, 0)

    # Set the position controller callback
    mj.set_mjcb_control(position_control)

    # Compute the RRT solution
    RRTQuery()

    # Launch the simulate viewer
    viewer.launch(model, data)
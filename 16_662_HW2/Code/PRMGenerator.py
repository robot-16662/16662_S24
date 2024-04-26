import Franka
import numpy as np
import random
import pickle
import RobotUtil as rt
import time

random.seed(13)

#Initialize robot object
mybot=Franka.FrankArm()

#Create environment obstacles - # these are blocks in the environment/scene (not part of robot) 
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

# Central block ahead of the robot
envpoints, envaxes = rt.BlockDesc2Points(rt.rpyxyz2H([0,0.,0.],[0.45, 0, 0.25]),[0.5,0.4,0.5])
pointsObs.append(envpoints), axesObs.append(envaxes)

prmVertices=[] # list of vertices
prmEdges=[] # adjacency list (undirected graph)
start = time.time()

#utility function for checking whether there are collisions along a path
def check_path(q1, q2, dq_check=0.05):
    dist = vertice_distance(q1,q2)
    for i in range(int(dist/dq_check)):
        q_check = q1 + i * dq_check * (q2 - q1) / dist
        if mybot.DetectCollision(q_check, pointsObs, axesObs):
            return False
    return True

#utility function for distance between two points
def vertice_distance(point1, point2):
    return np.sqrt(np.sum(np.square(point1 - point2)))

# Create PRM - generate collision-free vertices
def PRMGenerator():
    global prmVertices
    global prmEdges
    global pointsObs
    global axesObs
    
    pointsObs = np.array(pointsObs)
    axesObs = np.array(axesObs)

    des_vertice_count = 10000
    vertice_max_neighbor_dist = 1
    
    while len(prmVertices) < des_vertice_count:   #until we have our desired number of vertices
        # sample random poses

        q = np.array(mybot.SampleRobotConfig()) #get a random robot configuration
        if not mybot.DetectCollision(q, pointsObs, axesObs):    #if the configuration is valid
            prmVertices.append(q.tolist())  #add the configuration to the number of vertices

            q_neighbors = []
            for i in range(len(prmVertices) - 1):   #for each vertice in the map
                if vertice_distance(q, prmVertices[i]) < vertice_max_neighbor_dist: #if the vertice is close enough to be a neighbor
                    if check_path(q, prmVertices[i]):   #if the path to the the vertice is clear
                        q_neighbors.append(i)   #make the vertice a neighbor to the new node
                        prmEdges[i].append(len(prmVertices) - 1)    #make the new node a neighbor to the vertice
            prmEdges.append(q_neighbors)    #add the new edges to the edges list

        print(len(prmVertices)) #debug statement to ensure the road map is growing

    #Save the PRM such that it can be run by PRMQuery.py
    f = open("myPRM.p", 'wb')
    pickle.dump(prmVertices, f)
    pickle.dump(prmEdges, f)
    pickle.dump(pointsObs, f)
    pickle.dump(axesObs, f)
    f.close

if __name__ == "__main__":

    # Call the PRM Generator function and generate a graph
    PRMGenerator()

    print("\n", "Vertices: ", len(prmVertices),", Time Taken: ", time.time()-start)
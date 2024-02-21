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



#TODO: Create PRM - generate collision-free vertices
prmVertices=[]
prmEdges=[]
start = time.time()

#TODO: Fill in the following function using prmVertices and prmEdges to store
# the graph. The code at the end saves the graph into a python pickle file.

def PRMGenerator():

	while len(prmVertices)<1000:
		# sample random poses
		print(len(prmVertices))
		



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
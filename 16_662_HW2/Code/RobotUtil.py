import numpy as np 
import math

def rpyxyz2H(rpy,xyz):
	Ht=[[1,0,0,xyz[0]],
	    [0,1,0,xyz[1]],
            [0,0,1,xyz[2]],
            [0,0,0,1]]

	Hx=[[1,0,0,0],
	    [0,math.cos(rpy[0]),-math.sin(rpy[0]),0],
            [0,math.sin(rpy[0]),math.cos(rpy[0]),0],
            [0,0,0,1]]

	Hy=[[math.cos(rpy[1]),0,math.sin(rpy[1]),0],
            [0,1,0,0],
            [-math.sin(rpy[1]),0,math.cos(rpy[1]),0],
            [0,0,0,1]]

	Hz=[[math.cos(rpy[2]),-math.sin(rpy[2]),0,0],
            [math.sin(rpy[2]),math.cos(rpy[2]),0,0],
            [0,0,1,0],
            [0,0,0,1]]

	H=np.matmul(np.matmul(np.matmul(Ht,Hz),Hy),Hx)

	return H

def R2axisang(R):
	ang = math.acos(( R[0,0] + R[1,1] + R[2,2] - 1)/2)
	Z = np.linalg.norm([R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]])
	if Z==0:
		return[1,0,0], 0.
	x = (R[2,1] - R[1,2])/Z
	y = (R[0,2] - R[2,0])/Z
	z = (R[1,0] - R[0,1])/Z 	

	return[x, y, z], ang


def BlockDesc2Points(H, Dim):
	center = H[0:3,3]
	axes=[ H[0:3,0],H[0:3,1],H[0:3,2]]	

	corners=[
		center,
		center+(axes[0]*Dim[0]/2.)+(axes[1]*Dim[1]/2.)+(axes[2]*Dim[2]/2.),
		center+(axes[0]*Dim[0]/2.)+(axes[1]*Dim[1]/2.)-(axes[2]*Dim[2]/2.),
		center+(axes[0]*Dim[0]/2.)-(axes[1]*Dim[1]/2.)+(axes[2]*Dim[2]/2.),
		center+(axes[0]*Dim[0]/2.)-(axes[1]*Dim[1]/2.)-(axes[2]*Dim[2]/2.),
		center-(axes[0]*Dim[0]/2.)+(axes[1]*Dim[1]/2.)+(axes[2]*Dim[2]/2.),
		center-(axes[0]*Dim[0]/2.)+(axes[1]*Dim[1]/2.)-(axes[2]*Dim[2]/2.),
		center-(axes[0]*Dim[0]/2.)-(axes[1]*Dim[1]/2.)+(axes[2]*Dim[2]/2.),
		center-(axes[0]*Dim[0]/2.)-(axes[1]*Dim[1]/2.)-(axes[2]*Dim[2]/2.)
		]	
	# returns corners of BB and axes
	return corners, axes



def CheckPointOverlap(pointsA,pointsB,axis):	
	# TODO: check if sets of points projected on axis are overlapping

	return False


def CheckBoxBoxCollision(pointsA,axesA,pointsB,axesB):	
	#Sphere check
	
	if np.linalg.norm(pointsA[0]-pointsB[0])> (np.linalg.norm(pointsA[0]-pointsA[1])+np.linalg.norm(pointsB[0]-pointsB[1])):
		return False

	#TODO - SAT cuboid-cuboid collision check

	return True
	



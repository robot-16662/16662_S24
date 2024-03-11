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
    axes = [H[0:3,0],H[0:3,1],H[0:3,2]]	
 
    # find corners of the bounding box 3d using dimemsions and axes
    corners=[center,
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

def CheckPointOverlap(pointsA, pointsB, axis):	
    """
    Inputs:
        - pointsA: 9x3 array of points of box A
        - pointsB: 9x3 array of points of box B
        - axis: 3x1 array of axis to project points on
    
    Outputs:
        - overlap: boolean indicating if there is overlap
    
    """
    # TODO: Project both set of points on the axis and check for overlap

    raise NotImplementedError

def CheckBoxBoxCollision(pointsA, axesA, pointsB, axesB):
    """
    Inputs: 
        - pointsA: 9x3 array of points of box A
        - axesA: 3x3 array of axes of box A representing rotation matrix or direction vectors of surface normals
        - pointsB: 9x3 array of points of box B
        - axesB: 3x3 array of axes of box B representing rotation matrix or direction vectors of surface normals
    
    Outputs:
        - collision: boolean indicating if there is collision
    """	

    #Sphere check
    if np.linalg.norm(pointsA[0]-pointsB[0])> (np.linalg.norm(pointsA[0]-pointsA[1])+np.linalg.norm(pointsB[0]-pointsB[1])):
        return False

    #SAT cuboid-cuboid collision check. 
    #Hint: Use CheckPointOverlap() function to check for overlap along each axis
    
    #TODO: Check if cuboids collide along the surface normal of box A 
    #TODO: Check if cuboids collide along the surface normal of box B
    #TODO: Check for edge-edge collisions

    raise NotImplementedError

if __name__ == "__main__":
    # Run Test Cases
    test_origins = np.array( [[0,1,0], [1.5,-1.5,0], [0,0,-1], [3,0,0], [-1,0,-2], [1.8,0.5,1.5], [0,-1.2,0.4], [-0.8,0,-0.5]])
    test_ori = np.array([[0,0,0], [1,0,1.5], [0,0,0], [0,0,0], [.5,0,0.4], [-0.2,0.5,0], [0,0.785,0.785], [0,0,0.2]])
    test_dims = np.array([[0.8,0.8,0.8], [1,3,3], [2,3,1], [3,1,1], [2,0.7,2], [1,3,1], [1,1,1], [1,0.5,0.5]])

    ref_origin = np.array([0,0,0])
    ref_ori = np.array([0,0,0])
    ref_dims = np.array([3,1,2])

    ref_bbox_pts, ref_bbox_axis = BlockDesc2Points(rpyxyz2H(ref_ori, ref_origin), ref_dims)
    for i in range(len(test_origins)):
        Hi = rpyxyz2H(test_ori[i], test_origins[i])
        pts_i, ax_i = BlockDesc2Points(Hi, test_dims[i])
        ans_i = CheckBoxBoxCollision(np.array(ref_bbox_pts), np.array(ref_bbox_axis), np.array(pts_i), np.array(ax_i))
        print("Collision between block", i, "and given test block:\t", ans_i)  



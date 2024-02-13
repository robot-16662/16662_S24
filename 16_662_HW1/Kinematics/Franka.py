import numpy as np
import RobotUtil as rt
import math

class FrankArm:
    def __init__(self):
        # Robot descriptor taken from URDF file (rpy xyz for each rigid link transform) - NOTE: don't change
        self.Rdesc = [
            [0, 0, 0, 0., 0, 0.333],  # From robot base to joint1
            [-np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0, -0.316, 0],
            [np.pi/2, 0, 0, 0.0825, 0, 0],
            [-np.pi/2, 0, 0, -0.0825, 0.384, 0],
            [np.pi/2, 0, 0, 0, 0, 0],
            [np.pi/2, 0, 0, 0.088, 0, 0],
            [0, 0, 0, 0, 0, 0.107]  # From joint5 to end-effector center
        ]

        # Define the axis of rotation for each joint
        self.axis = [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 1]
        ]

        # Set base coordinate frame as identity - NOTE: don't change
        self.Tbase = [[1, 0, 0, 0],
                      [0, 1, 0, 0],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]]

        # Initialize matrices - NOTE: don't change this part
        self.Tlink = []  # Transforms for each link (const)
        self.Tjoint = []  # Transforms for each joint (init eye)
        self.Tcurr = []  # Coordinate frame of current (init eye)
        
        for i in range(len(self.Rdesc)):
            self.Tlink.append(rt.rpyxyz2H(
                self.Rdesc[i][0:3], self.Rdesc[i][3:6]))
            self.Tcurr.append(np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                              [0, 0, 1, 0.], [0, 0, 0, 1]]))
            self.Tjoint.append(np.array([[1, 0, 0, 0], [0, 1, 0, 0],
                               [0, 0, 1, 0.], [0, 0, 0, 1]]))

        self.Tlinkzero = rt.rpyxyz2H(self.Rdesc[0][0:3], self.Rdesc[0][3:6])

        self.Tlink[0] = np.matmul(self.Tbase, self.Tlink[0])

        # initialize Jacobian matrix
        self.J = np.zeros((6, 7))

        self.q = [0., 0., 0., 0., 0., 0., 0.]
        self.ForwardKin([0., 0., 0., 0., 0., 0., 0.])

    def ForwardKin(self, ang):
        '''
        inputs: joint angles
        outputs: joint transforms for each joint, Jacobian matrix
        '''

        self.q = ang

        # Compute current joint and end effector coordinate frames (self.Tjoint). Remember that not all joints rotate about the z axis!

        for i in range(0,len(self.Rdesc)):  #for each joint, calculate current Transform based on q
            self.Tjoint[i] = rt.rpyxyz2H(ang[i-1] * np.array(self.axis[i]), np.zeros(3))    #calculate joint transform
            if i == 0:
                self.Tcurr[i] = self.Tbase @ self.Tlink[0]  #on first joint, use base + link 0
            else:
                self.Tcurr[i] = self.Tcurr[i-1] @ self.Tjoint[i] @ self.Tlink[i]    #if not first joint, apply transform of joint and link to previous frame transform
        
        for i in range(len(self.q)):    #for each joint transform, pull info and put it into the Jacobian
            self.J[:,i] = np.array([rt.so3(self.Tcurr[i][0:3,2].reshape(3)) @ (self.Tcurr[-1][0:3,3] - self.Tcurr[i][0:3,3]), self.Tcurr[i][0:3,2]]).reshape(6)

        return self.Tcurr, self.J

    def IterInvKin(self, ang, TGoal, x_eps=1e-3, r_eps=1e-3):
        '''
        inputs: starting joint angles (ang), target end effector pose (TGoal)

        outputs: computed joint angles to achieve desired end effector pose, 
        Error in your IK solution compared to the desired target
        '''

        C = np.diag([1000000, 1000000, 1000000, 100, 100, 100]) #define C and W weight matrices
        W = np.diag([1,1,100,100,1,1,100])

        max_step = 0.01 #define max iteration step in the geometric space

        while(True):
            
            T_err = TGoal[0:3,3] - self.Tcurr[-1][0:3,3]    #get error in translation

            R_err = TGoal[0:3, 0:3] @ self.Tcurr[-1][0:3, 0:3].T    #get error in rotation
            aa_axis, aa_ang = rt.R2axisang(R_err)       #convert to axis angle
            rot_err_vector = np.array(aa_axis) * aa_ang

            err_vector = np.concatenate((T_err, rot_err_vector))    #concatenate the errors to get the error in the geometric space

            if(np.linalg.norm(T_err) < x_eps and np.linalg.norm(rot_err_vector) < r_eps):   #if converged, break
                break
            
            if(np.linalg.norm(err_vector) > max_step):  #limit size of error vector
                err_vector = err_vector / np.linalg.norm(err_vector) * max_step

            J_sharp = np.linalg.inv(self.J.T @ C @ self.J + W) @ self.J.T @ C   #compute the J# for damped least squares

            q = self.q + J_sharp @ err_vector   #perform damped least square interative step

            self.ForwardKin(q)  #update robot state using forward kinematics function
            
        return self.q, err_vector

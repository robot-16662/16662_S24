import mujoco as mj
from mujoco import viewer
import numpy as np
import math
import quaternion

# Set the XML filepath
xml_filepath = "../franka_emika_panda/panda_nohand_torque_fixed_board.xml"

force_error = []
times = []
forces = []

################################# Control Callback Definitions #############################

# Control callback for gravity compensation
def gravity_comp(model, data):
    # data.ctrl exposes the member that sets the actuator control inputs that participate in the
    # physics, data.qfrc_bias exposes the gravity forces expressed in generalized coordinates, i.e.
    # as torques about the joints
    
    data.ctrl[:7] = data.qfrc_bias[:7]

# Force control callback
def force_control(model, data):  # TODO:
    # Implement a force control callback here that generates a force of 15 N along the global x-axis,
    # i.e. the x-axis of the robot arm base. You can use the comments as prompts or use your own flow
    # of code. The comments are simply meant to be a reference.

    # Instantite a handle to the desired body on the robot
    body = data.body("hand")

    #define S for hybrid position-force control
    S = np.diag([0,1,1])

    # print(np.quaternion(body.xquat).as_euler_angles())

    #calculate the Jacobian
    J_pos = np.zeros((3,model.nv))
    J_rot = np.zeros((3,model.nv))
    mj.mj_jacBody(model,data,J_pos, J_rot, body.id)
    J_pos = J_pos[:,:7]

    #psuedoinverse of the jacobian
    J_pos_inv = J_pos.T @ np.linalg.inv(J_pos @ J_pos.T)

    # PD prameters for position control
    kp_pos = 100
    kd_pos = 0

    x_a = body.xpos     #current position
    x_d = np.array([0,0,0.6073])    #desired position
    x_e = x_d - x_a #position error
    x_es = S @ x_e  #use S to only affect dimensions in task space requiring position control
    q_es = J_pos_inv @ x_es #use inverse jacobian to get error in joint space

    tau_p = q_es * kp_pos + data.qvel[:7] * kd_pos  #calculate tau for position controller

    # PI paramters for force control
    kp_force = 1
    ki_force = 5
    # offset for the acceleration of gravity
    grav_offset = np.array([0, 0, 7.21547082])

    f_a = body.xmat.reshape((3,3)) @ data.sensordata - grav_offset  #current force from sensor
    f_d = np.array([15,0,0])    #desired force
    f_e = f_d - f_a     #force error
    f_es = (np.eye(3) - S) @ f_e    #use S to only affect dimensions in task space requiring force control
    force_error.append(f_es)    #keep history for integral control
    tau_es = J_pos.T @ f_es     #compute tau error using jacobian transpose
    tau_integral = J_pos.T @ np.trapz(np.array(force_error), dx=model.opt.timestep, axis=0)     #do the same for integral
    
    tau_f = tau_es * kp_force + tau_integral* ki_force  #calculate tau for force controller

    data.ctrl[:7] = data.qfrc_bias[:7] + tau_p + tau_f  #add position and force torques with gravity compensation

    # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION

    # Force readings updated here
    times.append(data.time)
    forces.append(data.sensordata[2])

# Control callback for an impedance controller
def impedance_control(model, data):  # TODO:

    # Implement an impedance control callback here that generates a force of 15 N along the global x-axis,
    # i.e. the x-axis of the robot arm base. You can use the comments as prompts or use your own flow
    # of code. The comments are simply meant to be a reference.

    # Instantite a handle to the desired body on the robot
    body = data.body("hand")

    f_d = 15    #desired force

    # PD parameters for impedance controller
    kp = 100
    kd = 1

    #calculate the Jacobian
    J_pos = np.zeros((3,model.nv))
    J_rot = np.zeros((3,model.nv))
    mj.mj_jacBody(model,data,J_pos, J_rot, body.id)
    J_pos = J_pos[:,:7]

    x_a = body.xpos     #current position
    x_d = np.array([x_a[0] + f_d / kp,0,0.6073])    #desired position, use f_d and kp to get a force of f_d in the x-direction at steady state
    x_e = x_d - x_a     #position error

    # Set the desired velocities
    v_a = J_pos @ data.qvel[:7]     #current velocity
    v_d = np.array([0,0,0])     #desired velocity
    v_e = v_d - v_a     #velocity error

    data.ctrl[:7] = data.qfrc_bias[:7] + J_pos.T @ (kd * v_e + kp * x_e)    #use PD control for impedance control, combined with gravity compensation

    # Set the control inputs

    # DO NOT CHANGE ANY THING BELOW THIS IN THIS FUNCTION

    # Update force sensor readings
    times.append(data.time)
    forces.append(data.sensordata[2])


def position_control(model, data):
    # Instantite a handle to the desired body on the robot
    body = data.body("hand")

    # Set the desired joint angle positions
    # desired_joint_positions = np.array(
    #     [0, 0, 0, -1.57079, 0, 1.57079, -0.7853])
    desired_joint_positions = np.array(
        [-0.51293991, 0.07921995, 0.52316144, -1.85316041, -3.11343046,
         2.79019664, -0.06636301])

    # Set the desired joint velocities
    desired_joint_velocities = np.array([0, 0, 0, 0, 0, 0, 0])

    # Desired gain on position error (K_p)
    Kp = 1000

    # Desired gain on velocity error (K_d)
    Kd = 1000

    # Set the actuator control torques
    data.ctrl[:7] = data.qfrc_bias[:7] + Kp * \
        (desired_joint_positions-data.qpos[:7]) + Kd * \
        (desired_joint_velocities-data.qvel[:7])


####################################### MAIN #####################################
if __name__ == "__main__":
    # Load the xml file here
    model = mj.MjModel.from_xml_path(xml_filepath)
    data = mj.MjData(model)

    # Set the simulation scene to the home configuration
    mj.mj_resetDataKeyframe(model, data, 0)

    ################################# Swap Callback Below This Line #################################
    # This is where you can set the control callback. Take a look at the Mujoco documentation for more
    # details. Very briefly, at every timestep, a user-defined callback function can be provided to
    # mujoco that sets the control inputs to the actuator elements in the model. The gravity
    # compensation callback has been implemented for you. Run the file and play with the model as
    # explained in the PDF

    mj.set_mjcb_control(position_control)  # TODO:

    ################################# Swap Callback Above This Line #################################

    # Launch the simulate viewer
    viewer.launch(model, data)

    # Save recorded force and time points as a csv file
    force = np.reshape(np.array(forces), (-1, 1))
    time = np.reshape(np.array(times), (-1, 1))
    plot = np.concatenate((time, force), axis=1)
    np.savetxt('force_vs_time.csv', plot, delimiter=',')

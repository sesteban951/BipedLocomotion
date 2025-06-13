#!/usr/bin/env python3

import numpy as np
import scipy as sp
import time
import yaml

from pydrake.all import *

# index for the state of the G1
class IDX:

    # generalized position indeces
    POS_X = 0  # BASE POSITION
    POS_Y = 1
    POS_Z = 2
    Q_X = 3    # BASE ORIENTATION
    Q_Y = 4
    Q_Z = 5
    Q_W = 6
    LHP = 7    # LEFT LEG
    LHR = 8   
    LHY = 9
    LKP = 10
    LAP = 11
    LAR = 12
    RHP = 13   # RIGHT LEG
    RHR = 14
    RHY = 15
    RKP = 16
    RAP = 17
    RAR = 18
    WAY = 19   # WAIST
    WAR = 20
    WAP = 21
    LSP = 22   # LEFT ARM
    LSR = 23
    LSY = 24
    LEP = 25
    LWR = 26   # LEFT WRIST
    LWP = 27
    LWY = 28
    RSP = 29   # RIGHT ARM
    RSR = 30
    RSY = 31
    REP = 32
    RWR = 33   # RIGHT WRIST
    RWP = 34
    RWY = 35

    # 12 DOF leg indeces
    idx_12dof = [Q_W, Q_X, Q_Y, Q_Z,            # base quat
                 POS_X, POS_Y, POS_Z,           # base position
                 LHP, LHR, LHY, LKP, LAP, LAR,  # left leg
                 RHP, RHR, RHY, RKP, RAP, RAR]  # right leg
    
    # full model
    idx_full = [Q_W, Q_X, Q_Y, Q_Z,           # base quat
                POS_X, POS_Y, POS_Z,          # base position
                LHP, LHR, LHY, LKP, LAP, LAR, # left leg
                RHP, RHR, RHY, RKP, RAP, RAR, # right leg
                LSP, LSR, LSY, LEP,           # left arm
                RSP, RSR, RSY, REP]           # right arm
    
    # base position indeces
    idx_base_pos = [POS_X, POS_Y, POS_Z]

    # base quaternion indeces
    idx_base_quat = [Q_W, Q_X, Q_Y, Q_Z]

    # leg position indeces
    idx_legs = [LHP, LHR, LHY, LKP, LAP, LAR,
                RHP, RHR, RHY, RKP, RAP, RAR]
    
    # waist positions
    idx_waist = [WAY, WAR, WAP]

    # arm positions
    idx_arms = [LSP, LSR, LSY, LEP, LWR, LWP, LWY,
                RSP, RSR, RSY, REP, RWR, RWP, RWY]

# main class for getting reference trajectories
class ReferenceTrajectory:

    def __init__(self, config):

        #  get the model type (full or half)
        self.model_type = config['model']['type']

        # load reference trajectory parameters
        reference_path = config['reference']['path']
        self.ref_dt = 1.0 / config['reference']['hz']

        # load the CSV file
        self.data = np.loadtxt(reference_path, delimiter=',', skiprows=1)
        self.ref_length = self.data.shape[0]          

        # class that holds the indeces
        self.idx = IDX()
        
        # extract the reference positions for the legs and base
        self.t_ref = self.create_reference_time_vector() # reference time
        
        # half model: 12 DOF
        if self.model_type == 'half':
            self.q_ref = self.data[:, self.idx.idx_12dof]  # reference positions        
        # full model: 36 DOF
        elif self.model_type == 'full':
            self.q_ref = self.data[:, self.idx.idx_full]  # reference positions
        else:
            raise ValueError("Unknown model type: {}".format(self.model_type))

        # MPC parameters
        # half model: 12 DOF
        if self.model_type == 'half':
            self.mpc_dt = config['MPC']['dt']                # MPC time step
            self.N = config['MPC']['num_steps'] + 1          # MPC horizon length
            self.q_horizon_ref = np.zeros((self.N, len(self.idx.idx_12dof)))     # MPC horizon positions
            self.v_horizon_ref = np.zeros((self.N, len(self.idx.idx_12dof)-1))   # MPC horizon velocities
        # full model: 36 DOF
        elif self.model_type == 'full':
            self.mpc_dt = config['MPC_full']['dt']           # MPC time step
            self.N = config['MPC_full']['num_steps'] + 1     # MPC horizon length
            self.q_horizon_ref = np.zeros((self.N, len(self.idx.idx_full)))     # MPC horizon positions
            self.v_horizon_ref = np.zeros((self.N, len(self.idx.idx_full)-1))   # MPC horizon velocities
        self.horizon = self.create_horizon_time_vector() # MPC horizon time vector
    
    # create a time vector for the whole reference trajectory
    def create_reference_time_vector(self):
        
        # get the size of the trajectory 
        rows = self.data.shape[0]
    
        # create a time vector of the same size
        integer_vec = np.arange(rows)
        time_vec = integer_vec * self.ref_dt

        return time_vec
    
    # create a time vector for MPC horizon
    def create_horizon_time_vector(self):
        
        # create time vector for the MPC horizon
        integer_vec = np.arange(self.N)
        time_vec = integer_vec * self.mpc_dt
        
        return time_vec

    # find interpolation intervals
    def get_interpolation_indeces(self, t_sim):

        # create the absolute time horizon
        horizon_sim = self.horizon + t_sim

        # for each time in the horizon, find the beginning and end indeces
        idx_interp = np.zeros((self.N, 2), dtype=int)   
        for i in range(self.N):
            t = horizon_sim[i]

            # Clip t to the time range, TODO: need to wrap around if t is outside the range
            if t <= self.t_ref[0]:
                idx_lower = 0
                idx_upper = 1
            elif t >= self.t_ref[-1]:
                idx_lower = len(self.t_ref) - 2
                idx_upper = len(self.t_ref) - 1
            else:
                idx_upper = np.searchsorted(self.t_ref, t, side='right')
                idx_lower = idx_upper - 1

            idx_interp[i, 0] = idx_lower
            idx_interp[i, 1] = idx_upper

        return idx_interp
    
    # interpolate the reference trajectory to get the full state trajectory
    def get_interpolated_trajectory(self, t_sim):

        # Get the interpolation index pairs for each horizon time point
        idx_pairs = self.get_interpolation_indeces(t_sim)  # shape (N, 2)

        # Get the absolute time for each step on the horizon
        horizon_sim = self.horizon + t_sim

        # Initialize the reference trajectory arrays
        for i in range(self.N):

            # Get the indices for the current horizon step
            idx_0, idx_1 = idx_pairs[i]
            t1 = self.t_ref[idx_0]
            t2 = self.t_ref[idx_1]
            q1 = self.q_ref[idx_0, :]
            q2 = self.q_ref[idx_1, :]

            # interpolate the state at the current horizon time
            t_interp = horizon_sim[i]

            # Perform interpolation
            self.q_horizon_ref[i, :] = self.interpolate(t_interp, t1, t2, q1, q2)

        # compute the velocities by finite differences
        for i in range(self.N - 1):
            q1 = self.q_horizon_ref[i, :]
            q2 = self.q_horizon_ref[i + 1, :]
            v_interp = self.finite_difference(q1, q2)
            self.v_horizon_ref[i, :] = v_interp

        # same velcoity for the last step
        self.v_horizon_ref[-1, :] = self.v_horizon_ref[-2, :]
        
        return self.q_horizon_ref, self.v_horizon_ref

    # linear interpolation
    def interpolate(self, t_sim, t1, t2, q1, q2):

        # ensure that time eval is in between t1 and t2
        assert (t1 <= t_sim <= t2), "Time evaluation must be between t1 and t2."
        
        # interpolate the state between two time points
        t_interp = t_sim - t1
        t_total = t2 - t1

        # cartesian interpolation 
        q1_cartesian = q1[4:] 
        q2_cartesian = q2[4:]
        q_cart_interp = q1_cartesian + (t_interp / t_total) * (q2_cartesian - q1_cartesian)

        # quaternion interpolation
        q1_quat = q1[:4]
        q2_quat = q2[:4]
        q_quat_interp = q1_quat + (t_interp / t_total) * (q2_quat - q1_quat)
        q_quat_interp /= np.linalg.norm(q_quat_interp)

        # stack the interpolated quaternion and cartesian positions
        q_interp = np.hstack((q_quat_interp, q_cart_interp))

        return q_interp
    
    # finite difference
    def finite_difference(self, q1, q2):

        # finite difference the base and leg positions
        q1_cartesian = q1[4:]
        q2_cartesian = q2[4:]
        v_cart = (q2_cartesian - q1_cartesian) / self.mpc_dt

        # finite difference the quaternion
        q1_quat = q1[:4] / np.linalg.norm(q1[:4])
        q2_quat = q2[:4] / np.linalg.norm(q2[:4])

        # Reorder to [x, y, z, w] for scipy
        q1_xyzw = np.array([q1_quat[1], q1_quat[2], q1_quat[3], q1_quat[0]])
        q2_xyzw = np.array([q2_quat[1], q2_quat[2], q2_quat[3], q2_quat[0]])

        # Use scipy to compute relative rotation
        r1 = sp.spatial.transform.Rotation.from_quat(q1_xyzw)
        r2 = sp.spatial.transform.Rotation.from_quat(q2_xyzw)
        r_rel = r2 * r1.inv()

        # Angular velocity in world frame (rotation vector / dt)
        omega_world = r_rel.as_rotvec() / self.mpc_dt

        # Rotate world-frame angular velocity into body frame at time q1
        omega_body = r1.inv().apply(omega_world)

        # Stack [angular vel in body frame, linear vel]
        v_finite_diff = np.hstack((omega_body, v_cart))

        return v_finite_diff

####################################################################################################


if __name__ == "__main__":

    # import the YAML config file
    config_path = "../config/config_g1.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # instantiate the reference trajectory class
    ref_traj = ReferenceTrajectory(config)

    # get the interpolated trajectory
    time_vector = ref_traj.horizon
    q_ref, _ = ref_traj.get_interpolated_trajectory(0.0)

    # entire reference trajectory
    # time_vector = ref_traj.t_ref
    # q_ref = ref_traj.q_ref
    # t_window = [0, 10]
    # time_mask = (time_vector >= t_window[0]) & (time_vector <= t_window[1])
    # time_vector = time_vector[time_mask]
    # q_ref = q_ref[time_mask, :]

    # start meshcat
    meshcat = StartMeshcat()

    # create a plant model
    if config['model']['type'] == 'half':
        model_file = config['model']['model_half']
    elif config['model']['type'] == 'full':
        model_file = config['model']['model_full']
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    models = Parser(plant).AddModels(model_file)
    plant.Finalize()

    # add default visualization
    AddDefaultVisualization(builder, meshcat)

    # build the diagram 
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    # start recording the meshcat playback
    meshcat.StartRecording()

    # start recording the meshcat playback
    time_elapsed = 0.0
    tot_time_des = time_vector[-1] - time_vector[0]
    dt = time_vector[1] - time_vector[0]
    for i in range(q_ref.shape[0]):

        # intialize the time for this iteration
        t0 = time.time()

        # Set the Drake model to have this state
        q0 = q_ref[i, :]
        plant.SetPositions(plant_context, q0)

        # Set the time in the Drake diagram. This will allow meshcat playback to work.
        time_elapsed += dt
        diagram_context.SetTime(time_elapsed)

        print("Playback time: {:.2f} s, Step: {:d}/{:d}".format(
            time_elapsed, i + 1, q_ref.shape[0]))    

        # Perform a forced publish event. This will propagate the plant's state to 
        # meshcat, without doing any physics simulation.
        diagram.ForcedPublish(diagram_context)

        # wait until the desired time has passed
        t1 = time.time()
        time_to_wait = t1 - t0
        if time_to_wait < dt:
            time.sleep(dt - time_to_wait)

    # Publish the meshcat recording
    meshcat.StopRecording()
    meshcat.PublishRecording()
#!/usr/bin/env python3

import numpy as np
import scipy as sp
import time
import yaml

from pydrake.all import *

import matplotlib.pyplot as plt

####################################################################################################

# main class for getting reference trajectories
class ReferenceTrajectory:

    def __init__(self, config):

        # MPC parameters
        self.mpc_dt = config['MPC']['dt']                # MPC time step
        self.N = config['MPC']['num_steps'] + 1          # MPC horizon length

        # MPC container for the horizon
        self.horizon = self.create_horizon_time_vector()
        self.q_horizon_ref = np.zeros((self.N, 23))   # MPC horizon positions
        self.v_horizon_ref = np.zeros((self.N, 22))   # MPC horizon velocities

        # load the CSV file, WARNING: fails when there is a trailing comma
        reference_path = config['reference_nodes']['path']
        self.T = config['reference_nodes']['T']
        self.node_data = np.loadtxt(reference_path, delimiter=',')

        # add a constant offset to the base
        z_offset = config['reference_nodes']['z_pos_offset']
        self.node_data[:, 6] += z_offset

        # create the reference trajectory
        self.t_ref, self.x_ref = self.create_reference_trajectory()
        self.ref_dt = self.t_ref[1] - self.t_ref[0]
        self.data = self.x_ref           # store the reference trajectory data

        self.t_ref = self.create_reference_time_vector()  
        self.q_ref = self.x_ref[:, :23]  # take the first 19 elements of the state vector

    # create a bezier curve for the reference trajectory
    def create_bezier_curve(self, x0, xf, t0, tf):

        # only take in the first 19 elements of the state vector
        # NOTE: ensure that the state vector is of the correct size
        x0 = np.array(x0).reshape(-1, 1)  # reshape to column vector
        xf = np.array(xf).reshape(-1, 1)  # reshape to column vector

        # create a bezier curve for the reference trajectory
        # control_pts = np.hstack([x0, (x0 + xf) / 2.0, xf])
        # control_pts = np.hstack([x0, x0, (x0 + xf) / 2.0, xf, xf])
        control_pts = np.hstack([x0, x0, x0, (x0 + xf) / 2.0, xf, xf, xf])
        # control_pts = np.hstack([x0, x0, x0, x0, (x0 + xf) / 2.0, xf, xf, xf, xf])

        # create a bezier curve
        curve = BezierCurve(t0, tf, control_pts)

        # # create an input vector for the spline
        T = tf - t0
        num_eval_pts = int(T / self.mpc_dt) + 1 # enforces uniform dt
        input = list(np.linspace(t0, tf, num=num_eval_pts))

        # evaluate the spline
        output = []
        for t in input:

            # evaluate the bezier curve at time t
            value = curve.value(t)

            # normalize the quaternion
            value[:4] /= np.linalg.norm(value[:4])

            # append the value to the output
            output.append(curve.value(t))

        return input, output
    
    # stitch together the reference trajectory
    def create_reference_trajectory(self):

        # number of configuration points and trajectories
        num_data_pts = self.node_data.shape[0]
        num_trajs = num_data_pts - 1

        # make sure that the number of periods in self.T is equal to the number of data points
        assert (num_data_pts-1) == len(self.T)

        # create list of periods
        T = 1.0
        integer_vec = np.arange(num_data_pts) * T

        # create the time vector for the reference trajectory
        integer_vec = np.zeros(len(self.T) + 1)
        T_now = 0.0
        for i in range(len(self.T)):
            integer_vec[i] = T_now
            T_now += self.T[i]
        integer_vec[-1] = T_now  # Add the final time

        # create a bezier curve for each trajectory
        t_ref_list = []
        x_ref_list = []
        for i in range(num_trajs):
            
            # get the first and last points of the trajectory
            t0 = integer_vec[i]
            tf = integer_vec[i + 1]
            x0 = self.node_data[i]
            xf = self.node_data[i + 1]

            # create a bezier curve for the trajectory
            t_list, x_list = self.create_bezier_curve(x0, xf, t0, tf)

            # remove the last point of the trajectory so that t_last = tf - dt
            if i != num_trajs - 1:
                t_list.pop()  # remove the last point
                x_list.pop()  # remove the last point

            # append the time and position vectors to the list
            t_ref_list.append(t_list)
            x_ref_list.append(x_list)

        # parse time lists
        self.t_ref = np.concatenate(t_ref_list).reshape(-1, 1)  # reshape to column vector

        # parse position lists
        for i in range(num_trajs):
            
            # get the intermediate trajectory
            traj = x_ref_list[i]

            # create a numpy array for the trajectory
            num_pts = len(traj)
            dim = traj[0].shape[0]  
            traj_array = np.zeros((num_pts, dim))

            # populate the numpy array with the trajectory
            for j in range(num_pts):
                traj_array[j, :] = traj[j].reshape(-1)

            # append the trajectory to the reference trajectory
            if i == 0:
                self.x_ref = traj_array
            else:
                self.x_ref = np.vstack((self.x_ref, traj_array))

        return self.t_ref, self.x_ref
    

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
            
            # Get the absolute time for the current horizon step
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

        # beyond the bounds of the reference trajectory, return the last state
        if (t_sim >= t2):
            q_interp = q2

        # before the reference trajectory, return the first state
        elif (t_sim <= t1):
            q_interp = q1
            
        # between two time points, interpolate the state
        else:
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

    # # entire reference trajectory
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
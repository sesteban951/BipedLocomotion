#!/usr/bin/env python3

import numpy as np
import scipy as sp
import time
import yaml

from pydrake.all import *

# import bezier
import matplotlib.pyplot as plt

####################################################################################################

# main class for getting reference trajectories
class ReferenceTrajectory:

    def __init__(self, config):

        # load the CSV file, WARNING: fails when there is a trailing comma
        reference_path = config['reference_nodes']['path']
        self.T = config['reference_nodes']['T'] 
        self.data = np.loadtxt(reference_path, delimiter=',')

        # MPC parameters
        self.mpc_dt = config['MPC']['dt']                # MPC time step
        self.N = config['MPC']['num_steps'] + 1          # MPC horizon length

        # MPC container for the horizon
        self.t_horizon = self.create_horizon_time_vector()
        self.q_horizon_ref = np.zeros((self.N, 19))     # MPC horizon positions
        self.v_horizon_ref = np.zeros((self.N, 18))   # MPC horizon velocities

        # create the reference trajectory
        self.create_reference_trajectory()

    # create a time vector for MPC horizon
    def create_horizon_time_vector(self):
        
        # create time vector for the MPC horizon
        integer_vec = np.arange(self.N)
        time_vec = integer_vec * self.mpc_dt
        
        return time_vec

    # create a bezier curve for the reference trajectory
    def create_bezier_curve(self, x0, xf, t0, tf):

        # only take in the first 19 elements of the state vector
        # NOTE: ensure that the state vector is of the correct size
        x0 = np.array(x0).reshape(-1, 1)  # reshape to column vector
        xf = np.array(xf).reshape(-1, 1)  # reshape to column vector

        # create a bezier curve for the reference trajectory
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
        num_data_pts = self.data.shape[0]
        num_trajs = num_data_pts - 1

        # create list of periods
        T = self.T
        integer_vec = np.arange(num_data_pts) * T
        
        # create a bezier curve for each trajectory
        t_ref_list = []
        x_ref_list = []
        for i in range(num_trajs):
            
            # get the first and last points of the trajectory
            t0 = integer_vec[i]
            tf = integer_vec[i + 1]
            x0 = self.data[i]
            xf = self.data[i + 1]

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


####################################################################################################


if __name__ == "__main__":

    # import the YAML config file
    config_path = "../config/config_g1.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # instantiate the reference trajectory class
    ref_traj = ReferenceTrajectory(config)

    # get the interpolated trajectory
    time_vector, x_ref = ref_traj.create_reference_trajectory()
    q_ref = x_ref[:, :19]  # take the first 19 elements of the state vector

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
    dt = time_vector[1][0] - time_vector[0][0]
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
#!/usr/bin/env python3

import numpy as np
import scipy as sp
import time
import yaml

# from pydrake.all import *

import bezier
import matplotlib.pyplot as plt

####################################################################################################

# main class for getting reference trajectories
class ReferenceTrajectory:

    def __init__(self, config):

        # TODO: somehow load the sequence of configurations from a file
        # q_list = 
        # T_list = 

        # MPC parameters
        self.mpc_dt = config['MPC']['dt']                # MPC time step
        self.N = config['MPC']['num_steps'] + 1          # MPC horizon length

        # MPC container for the horizon
        self.t_horizon = self.create_horizon_time_vector()
        self.q_horizon_ref = np.zeros((self.N, 19))     # MPC horizon positions
        self.v_horizon_ref = np.zeros((self.N, 18))   # MPC horizon velocities

        T = 2.0
        # create a time vector for the MPC horizon
        self.t_ref = self.create_time_spline_vector(T)

    # create a time vector for MPC horizon
    def create_horizon_time_vector(self):
        
        # create time vector for the MPC horizon
        integer_vec = np.arange(self.N)
        time_vec = integer_vec * self.mpc_dt
        
        return time_vec
        
    # create spline for time
    def create_time_spline_vector(self, T1, T2):

        # create a bezier curve for the time vector
        time_pts = np.array([[T1, T2]])
        nodes = np.asfortranarray(time_pts)
        curve = bezier.Curve(nodes, degree=1)

        # number of dt's that fit inside T2 - T1
        dT = T2 - T1
        num_dt = int(np.ceil(dT / self.mpc_dt))

        # create a time vector for the spline
        time_vector = np.linspace(T1, T2, num_dt + 1)

        # plot this curve
        time_curve = curve.evaluate_list(time_vector)
        
        plt.figure()
        plt.plot(time_vector, time_curve[0, :], label='Time Spline')
        # plt.xlabel('Time (s)')
        # plt.ylabel('Time (s)')
        # plt.title('Time Spline')
        

####################################################################################################


if __name__ == "__main__":

    # import the YAML config file
    config_path = "../config/config_g1.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # instantiate the reference trajectory class
    ref_traj = ReferenceTrajectory(config)

    # # get the interpolated trajectory
    # time_vector = ref_traj.horizon
    # q_ref, _ = ref_traj.get_interpolated_trajectory(0.0)

    # entire reference trajectory
    # time_vector = ref_traj.t_ref
    # q_ref = ref_traj.q_ref
    # t_window = [0, 10]
    # time_mask = (time_vector >= t_window[0]) & (time_vector <= t_window[1])
    # time_vector = time_vector[time_mask]
    # q_ref = q_ref[time_mask, :]

    # # start meshcat
    # meshcat = StartMeshcat()

    # # create a plant model
    # if config['model']['type'] == 'half':
    #     model_file = config['model']['model_half']
    # elif config['model']['type'] == 'full':
    #     model_file = config['model']['model_full']
    # builder = DiagramBuilder()
    # plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    # models = Parser(plant).AddModels(model_file)
    # plant.Finalize()

    # # add default visualization
    # AddDefaultVisualization(builder, meshcat)

    # # build the diagram 
    # diagram = builder.Build()
    # diagram_context = diagram.CreateDefaultContext()
    # plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    # # start recording the meshcat playback
    # meshcat.StartRecording()

    # # start recording the meshcat playback
    # time_elapsed = 0.0
    # tot_time_des = time_vector[-1] - time_vector[0]
    # dt = time_vector[1] - time_vector[0]
    # for i in range(q_ref.shape[0]):

    #     # intialize the time for this iteration
    #     t0 = time.time()

    #     # Set the Drake model to have this state
    #     q0 = q_ref[i, :]
    #     plant.SetPositions(plant_context, q0)

    #     # Set the time in the Drake diagram. This will allow meshcat playback to work.
    #     time_elapsed += dt
    #     diagram_context.SetTime(time_elapsed)

    #     print("Playback time: {:.2f} s, Step: {:d}/{:d}".format(
    #         time_elapsed, i + 1, q_ref.shape[0]))    

    #     # Perform a forced publish event. This will propagate the plant's state to 
    #     # meshcat, without doing any physics simulation.
    #     diagram.ForcedPublish(diagram_context)

    #     # wait until the desired time has passed
    #     t1 = time.time()
    #     time_to_wait = t1 - t0
    #     if time_to_wait < dt:
    #         time.sleep(dt - time_to_wait)

    # # Publish the meshcat recording
    # meshcat.StopRecording()
    # meshcat.PublishRecording()
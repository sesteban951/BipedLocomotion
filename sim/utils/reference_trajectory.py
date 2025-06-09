#!/usr/bin/env python3

import numpy as np
import scipy as sp
import time
import yaml

from pydrake.all import *

# index for the state of the G1
class IDX:

    # generalized position indeces
    POS_X = 0 # BASE POSITION
    POS_Y = 1
    POS_Z = 2
    Q_X = 3   # BASE ORIENTATION
    Q_Y = 4
    Q_Z = 5
    Q_W = 6
    LHP = 7   # LEFT LEG
    LHR = 8   
    LHY = 9
    LKP = 10
    LAP = 11
    LAR = 12
    RHP = 13  # RIGHT LEG
    RHR = 14
    RHY = 15
    RKP = 16
    RAP = 17
    RAR = 18
    WASIT_YAW = 19    # WAIST
    WAIST_ROLL = 20
    WAIST_PITCH = 21
    LSP = 22  # LEFT ARM
    LSR = 23
    LSY = 24
    LEP = 25
    LWR = 26  # LEFT WRIST
    LWP = 27
    LWY = 28
    RSP = 29  # RIGHT ARM
    RSR = 30
    RSY = 31
    REP = 32
    RWR = 33  # RIGHT WRIST
    RWP = 34
    RWY = 35

    # 12 DOF leg indeces
    idx_12dof = [Q_W, Q_X, Q_Y, Q_Z,
                 POS_X, POS_Y, POS_Z,
                 LHP, LHR, LHY, LKP, LAP, LAR,
                 RHP, RHR, RHY, RKP, RAP, RAR]

# main class for getting reference trajectories
class ReferenceTrajectory:

    def __init__(self, config):

        # load the model file from the config
        model_file = "../../models/g1_12dof_obj.urdf"
        plant = MultibodyPlant(0)
        Parser(plant).AddModels(model_file)
        plant.Finalize()

        # get the number of positions and velocities
        self.nq = plant.num_positions()
        self.nv = plant.num_velocities()

        # class that holds the indeces
        self.idx = IDX()

        # internal time and state parmeters
        self.t_current = 0.0
        self.q_current = np.zeros(plant.num_positions())    
        self.v_current = np.zeros(plant.num_velocities())

        # load the CSV file
        reference_path = config['reference']['path']
        self.dt = 1.0 / config['reference']['hz']
        self.data = self.load_reference_trajectory(reference_path)
        self.q_legs = self.data[:, self.idx.idx_12dof]  # reference positions
        self.ref_length = self.q_legs.shape[0]  # length of the reference trajectory

        # create a time vector
        self.time_vector = self.create_time_vector()

    # load data from a CSV file
    def load_reference_trajectory(self,reference_path):

        # load the CSV file
        data = np.loadtxt(reference_path, delimiter=',', skiprows=1)
        return data
    
    # create a time vector
    def create_time_vector(self):
        
        # get the size of the trajectory 
        rows = self.data.shape[0]
        
        # create a time vector of the same size
        integer_vec = np.arange(rows)
        time_vec = integer_vec * self.dt

        return time_vec

####################################################################################################


if __name__ == "__main__":

    # import the YAML config file
    config_path = "../config/config_g1.yaml"
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # instantiate the reference trajectory class
    ref_traj = ReferenceTrajectory(config)

    # get the time vector and the reference positions
    time_vector = ref_traj.time_vector
    q_ref = ref_traj.q_legs

    # start meshcat
    meshcat = StartMeshcat()

    # create a plant model
    model_file = "../../models/g1_12dof_obj.urdf"
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    models = Parser(plant).AddModels(model_file)

    plant.Finalize()

    AddDefaultVisualization(builder, meshcat)

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    time_elapsed = 0.0
    tot_time_des = time_vector[-1] - time_vector[0]
    configs_per_sec = ref_traj.ref_length / tot_time_des
    dt = 1.0 / configs_per_sec
    for i in range(ref_traj.ref_length):

        # Wait for the next state estimate  
        time.sleep(ref_traj.dt)

        # Set the Drake model to have this state
        q0 = q_ref[i, :]
        plant.SetPositions(plant_context, q0)

        # Set the time in the Drake diagram. This will allow meshcat playback to work.
        time_elapsed += dt
        diagram_context.SetTime(time_elapsed)

        # Perform a forced publish event. This will propagate the plant's state to 
        # meshcat, without doing any physics simulation.
        diagram.ForcedPublish(diagram_context)

    # Publish the meshcat recording
    meshcat.StopRecording()
    meshcat.PublishRecording()
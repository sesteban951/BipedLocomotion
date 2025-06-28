#!/usr/bin/env python

##
#
# 3D reference trajectory for the G1 humanoid model.
#
##

# import standard libraries
import time
import numpy as np
import yaml
import csv

# import the pydrake modules
from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    ApplyVisualizationConfig,
    VisualizationConfig,
    Parser,
    Box,
    RigidTransform,
    CoulombFriction,
    DiscreteContactApproximation,
    Simulator,
    JointActuatorIndex,
    PdControllerGains,
    VectorLogSink,
    MultibodyPlant,
    Sphere,
    Rgba
)

# import the pyidto modules
from pyidto import (
    TrajectoryOptimizer,
    SolverParameters,
    ProblemDefinition
)

# import the custom modules
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils'))
from mpc_utils import Interpolator, ModelPredictiveController # type: ignore
from reference_trajectory import ReferenceTrajectory          # type: ignore

# import the yaml config
config_path = "../config/config_g1_split_traj.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

#####################################################################################
    
def create_optimizer(model_file, q0):
    """
    Create a trajectory optimizer object that can be used for MPC.
    """

    # get the model type from the config
    mpc_type = 'MPC'
    
    # Create the system diagram that the optimizer uses
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=config[mpc_type]['dt'])
    Parser(plant).AddModels(model_file)

    plant.RegisterCollisionGeometry(
        plant.world_body(), 
        RigidTransform(p=[0, 0, -25]), 
        Box(50, 50, 50), "ground", 
        CoulombFriction(0.7, 0.7))

    plant.Finalize()
    diagram = builder.Build()

    nq = plant.num_positions()
    nv = plant.num_velocities()

    # Specify a cost function and target trajectory
    problem = ProblemDefinition()
    problem.num_steps = config[mpc_type]['num_steps']
    problem.q_init = np.copy(q0)
    problem.v_init = np.zeros(nv)
    
    # weights
    problem.Qq = np.diag(config[mpc_type]['Qq'])
    problem.Qv = np.diag(config[mpc_type]['Qv'])
    problem.R = np.diag(config[mpc_type]['R'])
    problem.Qf_q = config[mpc_type]['Qf_q_scaling'] * np.copy(problem.Qq)
    problem.Qf_v = config[mpc_type]['Qf_v_scaling'] * np.copy(problem.Qv)

    v_nom = np.zeros(nv)
    problem.q_nom = [np.copy(q0) for i in range(problem.num_steps + 1)]
    problem.v_nom = [np.copy(v_nom) for i in range(problem.num_steps + 1)]

    # Set the solver parameters
    params = SolverParameters()
    params.max_iterations = config[mpc_type]['max_iterations']
    params.scaling = config[mpc_type]['scaling']
    params.equality_constraints = config[mpc_type]['equality_constraints']
    params.Delta0 = config[mpc_type]['Delta0']
    params.Delta_max = config[mpc_type]['Delta_max']
    params.num_threads = config[mpc_type]['num_threads']
    params.contact_stiffness = config[mpc_type]['contact_stiffness']
    params.dissipation_velocity = config[mpc_type]['dissipation_velocity']
    params.smoothing_factor = config[mpc_type]['smoothing_factor']
    params.friction_coefficient = config[mpc_type]['friction_coefficient']
    params.stiction_velocity = config[mpc_type]['stiction_velocity']
    params.verbose = config[mpc_type]['verbose']

    # Create the optimizer
    optimizer = TrajectoryOptimizer(diagram, plant, problem, params)

    # Return the optimizer, along with the diangram and plant, which must
    # stay in scope along with the optimizer

    return optimizer, diagram, plant

#####################################################################################

class G1_MPC(ModelPredictiveController):
    """
    A Model Predictive Controller for the Achilles humanoid.
    """
    def __init__(self, optimizer, q_guess, mpc_rate, meshcat, reference_trajectory):
        ModelPredictiveController.__init__(self, optimizer, q_guess, 19, 18, mpc_rate)

        # create an internal plant for the controller
        model_path = config['model']['model_half']
        self.plant = MultibodyPlant(0.0)
        Parser(self.plant).AddModels(model_path)
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        # create internal reference trajecotry object
        self.reference_trajectory = reference_trajectory

        # current sim time
        self.t_sim = 0.0

        # for visualzing the COM
        self.meshcat = meshcat
        self.sphere_com = Sphere(0.015)
        self.red_color = Rgba(1.0, 0.0, 0.0, 1.0)
        self.meshcat.SetObject("com", self.sphere_com, self.red_color)

    def UpdateNominalTrajectory(self, context):
        """
        Shift the reference trajectory based on the current position.
        """

        # get the current sim time
        self.t_sim = context.get_time()

        # visualize the COM
        self.VisualizeCOM(context)

        # Update the reference trajectory
        q_nom, v_nom = self.reference_trajectory.get_interpolated_trajectory(self.t_sim)
        idx_traj = self.reference_trajectory.get_current_index_in_trajectory(self.t_sim)

        # print the sim time and the current trajectory index
        print(f"index: {idx_traj}, sim time: {self.t_sim:.4f}")

        self.optimizer.UpdateNominalTrajectory(q_nom, v_nom)

    # print the current state
    def VisualizeCOM(self, context):
        """
        Visualize the center of mass of the model instance.
        """
        # Get the current state
        x0 = self.state_input_port.Eval(context)
        q0 = x0[:self.nq]
        v0 = x0[self.nq:]

        # Set the positions and velocities in the internal plant
        self.plant.SetPositions(self.plant_context, q0)
        self.plant.SetVelocities(self.plant_context, v0)

        # Get the center of mass position projection onto ground
        com_pos = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context)
        com_pos[2] = 0.0  # project onto ground plane

        # plot it on meshcat
        self.meshcat.SetTransform("com", RigidTransform(com_pos), self.t_sim)

#####################################################################################


if __name__=="__main__":

    # start meshcat
    meshcat = StartMeshcat()

    # set up the config
    sim_time_step = config['sim']['time_step']
    ground_color = np.array([0.5, 0.5, 0.5, 1.0])
    Kp = np.array(config['gains']['Kp'])
    Kd = np.array(config['gains']['Kd'])
    mpc_rate = config['MPC']['mpc_rate']

    #############################################################################

    # initialize the reference trajectory
    ref_traj = ReferenceTrajectory(config)

    # intial configurations
    model_files = [config['model']['model_m4'],
                   config['model']['model_half']]
    
    # initial positions
    q0 = np.array(config['q0'])  # initial position
    v0 = np.zeros(18)  # initial velocity

    # start meshcat recording
    meshcat.StartRecording()

    # loop over the simulation runs
    for j in range(2):

        # simualtion segment 
        print(f"Running simulation segment {j}...")

        # create an object of the reference trajectory
        ref_traj = ReferenceTrajectory(config)

        # get the model file for this simulation segment
        model_file = model_files[j]

        # Set up a Drake diagram for simulation
        builder = DiagramBuilder()
        plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)
        plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
        models = Parser(plant).AddModels(model_file)  # robot model
    
        # Add ground
        plant.RegisterCollisionGeometry(  # ground
            plant.world_body(), 
            RigidTransform(p=[0, 0, -25]), 
            Box(50, 50, 50), "ground", 
            CoulombFriction(0.7, 0.7))
        plant.RegisterVisualGeometry(  # ground
            plant.world_body(), 
            RigidTransform(p=[0, 0, -25]), 
            Box(50, 50, 50), "ground", 
            ground_color)
        

        actuator_indices = [JointActuatorIndex(i) for i in range(plant.num_actuators())]
        for actuator_index, Kp_, Kd_ in zip(actuator_indices, Kp, Kd):
            plant.get_joint_actuator(actuator_index).set_controller_gains(
                PdControllerGains(p=Kp_, d=Kd_))    
        plant.Finalize()

        # Set up the trajectory optimization problem
        # Note that the diagram and plant must stay in scope while the optimizer is
        # being used
        optimizer, ctrl_diagram, ctrl_plant = create_optimizer(model_file, q0)
        q_guess = [q0 for _ in range(optimizer.num_steps() + 1)]

        # Create the MPC controller and interpolator systems
        controller = builder.AddSystem(G1_MPC(optimizer, q_guess, mpc_rate, meshcat, ref_traj))

        Bv = plant.MakeActuationMatrix()
        N = plant.MakeVelocityToQDotMap(plant.CreateDefaultContext())
        Bq = N@Bv
        interpolator = builder.AddSystem(Interpolator(Bq.T, Bv.T))
        
        # Wire the systems together
        builder.Connect(
            plant.get_state_output_port(), 
            controller.GetInputPort("state"))
        builder.Connect(
            controller.GetOutputPort("optimal_trajectory"), 
            interpolator.GetInputPort("trajectory"))
        builder.Connect(
            interpolator.GetOutputPort("control"), 
            plant.get_actuation_input_port())
        builder.Connect(
            interpolator.GetOutputPort("state"), 
            plant.get_desired_state_input_port(models[0])
        )

        # Logger state
        logger_state = builder.AddSystem(VectorLogSink(plant.num_positions() + plant.num_velocities()))
        builder.Connect(plant.get_state_output_port(), 
                        logger_state.get_input_port())
        
        # Logger applied torque
        logger_applied_torque = builder.AddSystem(VectorLogSink(plant.num_actuators()))
        builder.Connect(plant.get_net_actuation_output_port(), 
                        logger_applied_torque.get_input_port())
        
        # Logger commanded state
        logger_commanded_state = builder.AddSystem(VectorLogSink(plant.num_actuators() * 2))
        builder.Connect(interpolator.GetOutputPort("state"),
                        logger_commanded_state.get_input_port())
        
        # Logger torque feedforward 
        logger_torque_ff = builder.AddSystem(VectorLogSink(plant.num_actuators()))
        builder.Connect(interpolator.GetOutputPort("control"),
                        logger_torque_ff.get_input_port())
        
        # Connect the plant to meshcat for visualization
        vis_config = VisualizationConfig()
        vis_config.publish_contacts = config['contact_vis']
        ApplyVisualizationConfig(config=vis_config, builder=builder, meshcat=meshcat)

        # Build the system diagram
        diagram = builder.Build()
        diagram_context = diagram.CreateDefaultContext()
        plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

        # Set the initial state
        # q0 = initial_position(q0_idx)
        # v0 = np.array(config['v0'])

        plant.SetPositions(plant_context, q0)
        plant.SetVelocities(plant_context, v0)

        # Simulate and play back on meshcat
        
        simulator = Simulator(diagram, diagram_context)
        simulator.set_target_realtime_rate(config['sim']['real_time_rate'])
        simulator.AdvanceTo(config['sim']['duration'])
    
    meshcat.StopRecording()
    meshcat.PublishRecording()

    # # unpack the logged data
    # state_log = logger_state.FindLog(diagram_context)
    # applied_torque_log = logger_applied_torque.FindLog(diagram_context)
    # cmd_log = logger_commanded_state.FindLog(diagram_context)
    # torque_ff_log = logger_torque_ff.FindLog(diagram_context)

    # # unpack the logged data into numpy arrays
    # times = state_log.sample_times()
    # states = state_log.data().T
    # torques = applied_torque_log.data().T
    # commanded_states = cmd_log.data().T
    # torque_ff = torque_ff_log.data().T

    # # build the trajectory number vector
    # trajectory_indeces = np.arange(len(times)).reshape(-1, 1)
    # ref_traj = ReferenceTrajectory(config)
    # for i in range(len(times)):
    #     traj_idx = ref_traj.get_current_index_in_trajectory(times[i])
    #     trajectory_indeces[i] = traj_idx

    # # parse the state data
    # base_quat_w_actual = states[:, :4]
    # base_pos_w_actual = states[:, 4:7]
    # q_joint_target = commanded_states[:, :12]
    # v_joint_target = commanded_states[:, 12:]

    # # for every trajectory remove the first row
    # times = times[1:]
    # states = states[1:, :]
    # base_quat_w_actual = base_quat_w_actual[1:, :]
    # base_pos_w_actual = base_pos_w_actual[1:, :]
    # q_joint_target = q_joint_target[1:, :]
    # v_joint_target = v_joint_target[1:, :]
    # torque_ff = torque_ff[1:, :]
    # trajectory_indeces = trajectory_indeces[1:, :]

    # # save the state data to CSV files
    # save_folder = "./data/data/"

    # times_label = save_folder + "times.csv"
    # with open(times_label, mode='w') as file:
    #     writer = csv.writer(file)
    #     for i in range(len(times)):
    #         writer.writerow([times[i]])

    # full_state_label = save_folder + "full_state.csv"
    # with open(full_state_label, mode='w') as file:
    #     writer = csv.writer(file)
    #     for i in range(len(states)):
    #         writer.writerow(states[i])

    # base_quat_w_actual_label = save_folder + "base_quat_w_actual.csv"
    # with open(base_quat_w_actual_label, mode='w') as file:
    #     writer = csv.writer(file)
    #     for i in range(len(base_quat_w_actual)):
    #         writer.writerow(base_quat_w_actual[i])

    # base_pos_w_actual_label = save_folder + "base_pos_w_actual.csv"
    # with open(base_pos_w_actual_label, mode='w') as file:
    #     writer = csv.writer(file)
    #     for i in range(len(base_pos_w_actual)):
    #         writer.writerow(base_pos_w_actual[i])

    # q_joint_target_label = save_folder + "q_joint_target.csv"
    # with open(q_joint_target_label, mode='w') as file:
    #     writer = csv.writer(file)
    #     for i in range(len(q_joint_target)):
    #         writer.writerow(q_joint_target[i])

    # v_joint_target_label = save_folder + "v_joint_target.csv"
    # with open(v_joint_target_label, mode='w') as file:
    #     writer = csv.writer(file)
    #     for i in range(len(v_joint_target)):
    #         writer.writerow(v_joint_target[i])
    
    # torque_ffs_label = save_folder + "torques_ff.csv"
    # with open(torque_ffs_label, mode='w') as file:
    #     writer = csv.writer(file)
    #     for i in range(len(torque_ff)):
    #         writer.writerow(torque_ff[i])

    # traj_idx_label = save_folder + "trajectory_idx.csv"
    # with open(traj_idx_label, mode='w') as file:
    #     writer = csv.writer(file)
    #     for i in range(len(trajectory_indeces)):
    #         writer.writerow(trajectory_indeces[i])

    # print("Saved data to CSV files.")
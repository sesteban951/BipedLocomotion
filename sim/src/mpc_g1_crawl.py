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
    VectorLogSink
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
from reference_interpolation import ReferenceTrajectory       # type: ignore

# import the yaml config
config_path = "../config/config_g1.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

########################################################################################

class IDX():
    """
    Class to hold the indices of the model instance.
    """
    def __init__(self):
        
        # generalized positions
        self.QUAT_W = 0
        self.QUAT_X = 1
        self.QUAT_Y = 2
        self.QUAT_Z = 3
        self.POS_X = 4
        self.POS_Y = 5
        self.POS_Z = 6
        self.POS_LHP = 7
        self.POS_LHR = 8
        self.POS_LHY = 9
        self.POS_LKP = 10
        self.POS_LAP = 11
        self.POS_LAR = 12
        self.POS_RHP = 13
        self.POS_RHR = 14
        self.POS_RHY = 15
        self.POS_RKP = 16
        self.POS_RAP = 17
        self.POS_RAR = 18
        self.POS_LSP = 19
        self.POS_LEP = 20
        self.POS_RSP = 21
        self.POS_REP = 22

        # generalized velocities
        self.ANG_X = 0
        self.ANG_Y = 1
        self.ANG_Z = 2
        self.VEL_X = 3
        self.VEL_Y = 4
        self.VEL_Z = 5
        self.VEL_LHP = 6
        self.VEL_LHR = 7
        self.VEL_LHY = 8
        self.VEL_LKP = 9
        self.VEL_LAP = 10
        self.VEL_LAR = 11
        self.VEL_RHP = 12
        self.VEL_RHR = 13
        self.VEL_RHY = 14
        self.VEL_RKP = 15
        self.VEL_RAP = 16
        self.VEL_RAR = 17
        self.VEL_LSP = 18
        self.VEL_LEP = 19
        self.VEL_RSP = 20
        self.VEL_REP = 21

        # joints
        self.JOINT_LHP = 0
        self.JOINT_LHR = 1
        self.JOINT_LHY = 2
        self.JOINT_LKP = 3
        self.JOINT_LAP = 4
        self.JOINT_LAR = 5
        self.JOINT_RHP = 6
        self.JOINT_RHR = 7
        self.JOINT_RHY = 8
        self.JOINT_RKP = 9
        self.JOINT_RAP = 10
        self.JOINT_RAR = 11
        self.JOINT_LSP = 12
        self.JOINT_LEP = 13
        self.JOINT_RSP = 14
        self.JOINT_REP = 15

#####################################################################################

def standing_position():
    """
    Return a reasonable default standing position for the Achilles humanoid. 
    """
    return np.array(config['q0_crawl'])
    
    
def create_optimizer(model_file):
    """
    Create a trajectory optimizer object that can be used for MPC.
    """

    # get the model type from the config
    mpc_type = 'MPC_crawl'
    
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

    q_stand = standing_position()

    # Specify a cost function and target trajectory
    problem = ProblemDefinition()
    problem.num_steps = config[mpc_type]['num_steps']
    problem.q_init = np.copy(q_stand)
    problem.v_init = np.zeros(nv)
    
    # weights
    problem.Qq = np.diag(config[mpc_type]['Qq'])
    problem.Qv = np.diag(config[mpc_type]['Qv'])
    problem.R = np.diag(config[mpc_type]['R'])
    problem.Qf_q = config[mpc_type]['Qf_q_scaling'] * np.copy(problem.Qq)
    problem.Qf_v = config[mpc_type]['Qf_v_scaling'] * np.copy(problem.Qv)

    v_nom = np.zeros(nv)
    problem.q_nom = [np.copy(q_stand) for i in range(problem.num_steps + 1)]
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
    def __init__(self, optimizer, q_guess, mpc_rate):
        ModelPredictiveController.__init__(self, optimizer, q_guess, 23, 22, mpc_rate)

        # instantiate the model instance indices
        self.idx = IDX()

        # create internal reference trajecotry object
        self.reference_trajectory = ReferenceTrajectory(config)

        # current sim time
        self.t_sim = 0.0

    def UpdateNominalTrajectory(self, context):
        """
        Shift the reference trajectory based on the current position.
        """

        # Get the current state
        x0 = self.state_input_port.Eval(context)
        q0 = x0[:self.nq]
        v0 = x0[self.nq:]

        #  get the current sim time
        self.t_sim = context.get_time()

        print(f"Current sim time: {self.t_sim:.4f}")

        # Get the current nominal trajectory
        prob = self.optimizer.prob()
        q_nom = prob.q_nom
        v_nom = prob.v_nom

        # Shift the nominal trajectory
        dt = self.optimizer.time_step()
        vx = 0.25
        for i in range(self.num_steps + 1):
            q_nom[i][self.idx.POS_X] = q0[self.idx.POS_X] + vx * i * dt
            v_nom[i][self.idx.VEL_X] = vx

        # Update the reference trajectory
        # q_nom, v_nom = self.reference_trajectory.get_interpolated_trajectory(self.t_sim)

        self.optimizer.UpdateNominalTrajectory(q_nom, v_nom)

#####################################################################################


if __name__=="__main__":

    # start meshcat
    meshcat = StartMeshcat()

    # set up the model file
    model_file = config['model']['model_crawl']

    # Set up a Drake diagram for simulation
    builder = DiagramBuilder()
    sim_time_step = config['sim']['time_step']
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=sim_time_step)
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)
    models = Parser(plant).AddModels(model_file)  # robot model
   
    # Add ground
    ground_color = np.array(config['color']['ground'])
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

    # Add implicit PD controllers (must use kLagged or kSimilar)
    Kp = np.array(config['gains_crawl']['Kp'])
    Kd = np.array(config['gains_crawl']['Kd'])

    actuator_indices = [JointActuatorIndex(i) for i in range(plant.num_actuators())]
    for actuator_index, Kp, Kd in zip(actuator_indices, Kp, Kd):
        plant.get_joint_actuator(actuator_index).set_controller_gains(
            PdControllerGains(p=Kp, d=Kd))    
    plant.Finalize()

    # Set up the trajectory optimization problem
    # Note that the diagram and plant must stay in scope while the optimizer is
    # being used
    optimizer, ctrl_diagram, ctrl_plant = create_optimizer(model_file)
    q_guess = [standing_position() for _ in range(optimizer.num_steps() + 1)]

    # Create the MPC controller and interpolator systems
    mpc_rate = config['MPC_crawl']['mpc_rate']
    controller = builder.AddSystem(G1_MPC(optimizer, q_guess, mpc_rate))

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
    
    # Logger torque
    logger_torque = builder.AddSystem(VectorLogSink(plant.num_actuators()))
    builder.Connect(plant.get_net_actuation_output_port(), 
                    logger_torque.get_input_port())
    
    # Connect the plant to meshcat for visualization
    vis_config = VisualizationConfig()
    vis_config.publish_contacts = config['contact_vis']
    ApplyVisualizationConfig(config=vis_config, builder=builder, meshcat=meshcat)

    # Build the system diagram
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    # Set the initial state
    q0 = standing_position()
    v0 = np.array(config['v0_crawl'])

    plant.SetPositions(plant_context, q0)
    plant.SetVelocities(plant_context, v0)

    # Simulate and play back on meshcat
    meshcat.StartRecording()
    st = time.time()
    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(config['sim']['real_time_rate'])
    simulator.AdvanceTo(config['sim']['duration'])
    wall_time = time.time() - st
    print(f"sim time: {simulator.get_context().get_time():.4f}, "
          f"wall time: {wall_time:.4f}")
    meshcat.StopRecording()
    meshcat.PublishRecording()

    # unpack the logged data
    state_log = logger_state.FindLog(diagram_context)
    torque_log = logger_torque.FindLog(diagram_context)

    times = state_log.sample_times()
    states = state_log.data().T
    torques = torque_log.data().T

    # save the state data to CSV files
    save_folder = "./data/"

    times_label = save_folder + "times.csv"
    with open(times_label, mode='w') as file:
        writer = csv.writer(file)
        for i in range(len(times)):
            writer.writerow([times[i]])

    states_label = save_folder + "states.csv"
    with open(states_label, mode='w') as file:
        writer = csv.writer(file)
        for i in range(len(states)):
            writer.writerow(states[i])

    torques_label = save_folder + "torques.csv"
    with open(torques_label, mode='w') as file:
        writer = csv.writer(file)
        for i in range(len(torques)):
            writer.writerow(torques[i])
            
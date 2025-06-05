#!/usr/bin/env python

##
#
# MPC with 3D+arms version of the G1 humanoid.
#
##

# import standard libraries
import time
import numpy as np
import csv
import yaml

# import the pydrake modules
from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    ApplyVisualizationConfig,
    VisualizationConfig,
    LeafSystem,
    Parser,
    Box,
    RigidTransform,
    CoulombFriction,
    DiscreteContactApproximation,
    Simulator,
    JointActuatorIndex,
    PdControllerGains,
    BasicVector,
    MultibodyPlant,
    VectorLogSink,
    RollPitchYaw,
    RotationMatrix,
    Rgba, Sphere, Cylinder
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
from joystick import GamepadCommand                           # type: ignore

# import the yaml config
config_path = "../config/config_g1.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

#####################################################################################

def standing_position():
    """
    Return a reasonable default standing position for the Achilles humanoid. 
    """
    q_standing = np.array(config['q0'])
    return q_standing

def create_optimizer(model_file):
    """
    Create a trajectory optimizer object that can be used for MPC.
    """

    # Create the system diagram that the optimizer uses
    builder = DiagramBuilder()
    plant, _ = AddMultibodyPlantSceneGraph(builder, time_step=config['MPC']['dt'])
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
    problem.num_steps = config['MPC']['num_steps']
    problem.q_init = np.copy(q_stand)
    problem.v_init = np.zeros(nv)
    
    # weights
    problem.Qq = np.diag(config['MPC']['Qq'])
    problem.Qv = np.diag(config['MPC']['Qv'])
    problem.R = np.diag(config['MPC']['R'])
    problem.Qf_q = config['MPC']['Qf_q_scaling'] * np.copy(problem.Qq)
    problem.Qf_v = config['MPC']['Qf_v_scaling'] * np.copy(problem.Qv)

    v_nom = np.zeros(nv)
    problem.q_nom = [np.copy(q_stand) for i in range(problem.num_steps + 1)]
    problem.v_nom = [np.copy(v_nom) for i in range(problem.num_steps + 1)]

    # Set the solver parameters
    params = SolverParameters()
    params.max_iterations = config['MPC']['max_iterations']
    params.scaling = config['MPC']['scaling']
    params.equality_constraints = config['MPC']['equality_constraints']
    params.Delta0 = config['MPC']['Delta0']
    params.Delta_max = config['MPC']['Delta_max']
    params.num_threads = config['MPC']['num_threads']
    params.contact_stiffness = config['MPC']['contact_stiffness']
    params.dissipation_velocity = config['MPC']['dissipation_velocity']
    params.smoothing_factor = config['MPC']['smoothing_factor']
    params.friction_coefficient = config['MPC']['friction_coefficient']
    params.stiction_velocity = config['MPC']['stiction_velocity']
    params.verbose = config['MPC']['verbose']

    # Create the optimizer
    optimizer = TrajectoryOptimizer(diagram, plant, problem, params)

    # Return the optimizer, along with the diangram and plant, which must
    # stay in scope along with the optimizer

    return optimizer, diagram, plant

#####################################################################################


if __name__=="__main__":

    # start meshcat
    meshcat = StartMeshcat()

    # set up the model file
    model_file = "../../models/g1_12dof_obj.urdf"

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
    Kp = np.array(config['gains']['Kp'])
    Kd = np.array(config['gains']['Kd'])    
    actuator_indices = [JointActuatorIndex(i) for i in range(plant.num_actuators())]
    for actuator_index, Kp, Kd in zip(actuator_indices, Kp, Kd):
        plant.get_joint_actuator(actuator_index).set_controller_gains(
            PdControllerGains(p=Kp, d=Kd))    
    plant.Finalize()

    # # Set up the trajectory optimization problem
    # # Note that the diagram and plant must stay in scope while the optimizer is
    # # being used
    # optimizer, ctrl_diagram, ctrl_plant = create_optimizer(model_file)
    # q_guess = [standing_position() for _ in range(optimizer.num_steps() + 1)]

    # # add the joystick
    # joystick = builder.AddSystem(GamepadCommand(deadzone=0.05))
    
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
    v0 = np.array(config['v0'])
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

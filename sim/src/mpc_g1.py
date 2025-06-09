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

class G1_MPC(ModelPredictiveController):
    """
    A Model Predictive Controller for the Achilles humanoid.
    """
    def __init__(self, optimizer, q_guess, mpc_rate):
        ModelPredictiveController.__init__(self, optimizer, q_guess, 19, 18, mpc_rate)

    def UpdateNominalTrajectory(self, context):
        """
        Shift the reference trajectory based on the current position.
        """
        # instantiate the model instance indices
        idx = IDX()

        # Get the current state
        x0 = self.state_input_port.Eval(context)
        q0 = x0[:self.nq]
        v0 = x0[self.nq:]

        # Get the current nominal trajectory
        prob = self.optimizer.prob()
        q_nom = prob.q_nom
        v_nom = prob.v_nom

        # Shift the nominal trajectory
        dt = self.optimizer.time_step()
        vx = 0.25
        for i in range(self.num_steps + 1):
            q_nom[i][idx.POS_X] = q0[idx.POS_X] + vx * i * dt
            v_nom[i][idx.VEL_X] = vx

        self.optimizer.UpdateNominalTrajectory(q_nom, v_nom)


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
    optimizer, ctrl_diagram, ctrl_plant = create_optimizer(model_file)
    q_guess = [standing_position() for _ in range(optimizer.num_steps() + 1)]

    # Create the MPC controller and interpolator systems
    mpc_rate = config['MPC']['mpc_rate']
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

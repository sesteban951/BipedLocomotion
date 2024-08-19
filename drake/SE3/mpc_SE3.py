#!/usr/bin/env python

##
#
# MPC with a planar version of the Achilles humanoid.
#
##

from pydrake.all import (
    StartMeshcat,
    DiagramBuilder,
    AddMultibodyPlantSceneGraph,
    AddDefaultVisualization,
    Parser,
    Box,
    RigidTransform,
    CoulombFriction,
    DiscreteContactApproximation,
    Simulator,
    JointActuatorIndex,
    PdControllerGains,
    BasicVector,
    MultibodyPlant
)

import time
import numpy as np
import csv

from pyidto import (
    TrajectoryOptimizer,
    TrajectoryOptimizerSolution,
    TrajectoryOptimizerStats,
    SolverParameters,
    ProblemDefinition,
    FindIdtoResource,
)

import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from mpc_utils import Interpolator, ModelPredictiveController
from joystick import GamepadCommand

#--------------------------------------------------------------------------------------------------------------------------#

def standing_position():
    """
    Return a reasonable default standing position for the Achilles humanoid. 
    """
    # hip yaw, roll, pitch, 
    # knee, ankle
    # no arms (parallel legs, bent knees)
    # q_stand = np.array([
    #     1.0000, 0.0000, 0.0000, 0.0000,           # base orientation, (w, x, y, z)
    #     0.0000, 0.0000, 0.9300,                   # base position, (x,y,z)
    #     0.0000, 0.0209,-0.5515, 1.0239,-0.4725,   # left leg, (hip yaw, hip roll, hip pitch, knee, ankle) 
    #     0.0000,-0.0209,-0.3200, 0.9751,-0.6552,   # right leg, (hip yaw, hip roll, hip pitch, knee, ankle)
    # ])

    q_stand = np.array([
        1.0000, 0.0000, 0.0000, 0.0000,            # base orientation, (w, x, y, z)
        0.0000, 0.0000, 0.9300,                    # base position, (x,y,z)
        0.0000,  0.0209, -0.5515, 1.0239,-0.4725,  # left leg, (hip yaw, hip roll, hip pitch, knee, ankle) 
        0.0000, -0.0209, -0.5515, 1.0239,-0.4725,  # left leg, (hip yaw, hip roll, hip pitch, knee, ankle) 
    ])

    return q_stand

def create_optimizer(model_file):
    """
    Create a trajectory optimizer object that can be used for MPC.
    """

    # Create the system diagram that the optimizer uses
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.05)
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
    problem.num_steps = 50
    problem.q_init = np.copy(q_stand)
    problem.v_init = np.zeros(nv)
    
    # no arms
    problem.Qq = np.diag([
        10.0, 10.0, 10.0, 10.0,   # base orientation
        10.0, 10.0, 10.0,         # base position
        0.1, 0.1, 0.1, 0.1, 0.1,  # left leg
        0.1, 0.1, 0.1, 0.1, 0.1,  # right leg
    ])
    problem.Qv = 0.01 * np.eye(nv)
    problem.R = 0.01 * np.diag([
        100.0, 100.0, 100.0, 100.0,    # base orientation
        100.0, 100.0, 100.0,           # base position
        0.01, 0.01, 0.01, 0.01, 0.01,  # left leg
        0.01, 0.01, 0.01, 0.01, 0.01,  # right leg
    ])
    problem.Qf_q = 10.0 * np.copy(problem.Qq)
    problem.Qf_v = 1.0 * np.copy(problem.Qv)

    v_nom = np.zeros(nv)
    problem.q_nom = [np.copy(q_stand) for i in range(problem.num_steps + 1)]
    problem.v_nom = [np.copy(v_nom) for i in range(problem.num_steps + 1)]

    # Set the solver parameters
    params = SolverParameters()
    params.max_iterations = 1
    params.scaling = True
    params.equality_constraints = False
    params.Delta0 = 1e1
    params.Delta_max = 1e5
    params.num_threads = 10
    params.contact_stiffness = 10_000
    params.dissipation_velocity = 0.1
    params.smoothing_factor = 0.01
    params.friction_coefficient = 0.5
    params.stiction_velocity = 0.2
    params.verbose = False

    # Create the optimizer
    optimizer = TrajectoryOptimizer(diagram, plant, problem, params)

    # Return the optimizer, along with the diangram and plant, which must
    # stay in scope along with the optimizer
    return optimizer, diagram, plant

#--------------------------------------------------------------------------------------------------------------------------#

class AchillesMPC(ModelPredictiveController):
    """
    A Model Predictive Controller for the Achilles humanoid.
    """
    def __init__(self, optimizer, q_guess, mpc_rate, model_file):

        # inherit from the ModelPredictiveController class
        ModelPredictiveController.__init__(self, optimizer, q_guess, 17, 16, mpc_rate)

        self.joystick_port = self.DeclareVectorInputPort("joy_command",
                                                         BasicVector(5))  # LS_x, LS_y, RS_x, A button, RT (Xbox)
        
        self.v_max = 0.2
        self.q_stand = standing_position()

    def UpdateNominalTrajectory(self, context):
        """
        Shift the reference trajectory based on the current position.
        """
        # Get the current state
        x0 = self.state_input_port.Eval(context)
        q0 = x0[:self.nq]
        v0 = x0[self.nq:]

        # unpack the joystick commands
        joy_command = self.joystick_port.Eval(context)
        vx_des =  joy_command[1] * self.v_max
        vy_des = -joy_command[0] * self.v_max

        # Quit if the robot has fallen down
        base_height = q0[6]
        assert base_height > 0.0, "Oh no, the robot fell over!"

        # Get the current nominal trajectory
        prob = self.optimizer.prob()
        q_nom = prob.q_nom
        v_nom = prob.v_nom

        # Shift the nominal trajectory
        dt = self.optimizer.time_step()
        for i in range(self.num_steps + 1):
            q_nom[i][4] = q0[4] + vx_des * i * dt
            q_nom[i][5] = q0[5] + vy_des * i * dt
            v_nom[i][4] = vx_des
            v_nom[i][5] = vy_des

        self.optimizer.UpdateNominalTrajectory(q_nom, v_nom)

############################################################################################################################

if __name__=="__main__":

    meshcat = StartMeshcat()
    model_file = "../../models/achilles_SE3_drake.urdf"

    # Set up a Drake diagram for simulation
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=5e-3)
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)

    models = Parser(plant).AddModels(model_file)  # robot model
    
    plant.RegisterCollisionGeometry(  # ground
        plant.world_body(), 
        RigidTransform(p=[0, 0, -25]), 
        Box(50, 50, 50), "ground", 
        CoulombFriction(0.5, 0.5))

    # Add implicit PD controllers (must use kLagged or kSimilar)
    kp_hip = 750
    kp_knee = 750
    kp_ankle = 150
    kd_hip = 10
    kd_knee = 10
    kd_ankle = 2
    Kp = np.array([kp_hip, kp_hip, kp_hip, kp_knee, kp_ankle, kp_hip, kp_hip, kp_hip, kp_knee, kp_ankle])
    Kd = np.array([kd_hip, kd_hip, kd_hip, kd_knee, kd_ankle, kd_hip, kd_hip, kd_hip, kd_knee, kd_ankle])
    
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

    # add the joystick
    joystick = builder.AddSystem(GamepadCommand(deadzone=0.05))

    # Create the MPC controller and interpolator systems
    mpc_rate = 50  # Hz
    controller = builder.AddSystem(AchillesMPC(optimizer, q_guess, mpc_rate, model_file))

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
    builder.Connect(
        joystick.get_output_port(), 
        controller.GetInputPort("joy_command")
    )
    
    # Connect the plant to meshcat for visualization
    AddDefaultVisualization(builder, meshcat)

    # Build the system diagram
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    # Set the initial state
    q0 = standing_position()
    v0 = np.zeros(plant.num_velocities())
    plant.SetPositions(plant_context, q0)
    plant.SetVelocities(plant_context, v0)

    # Simulate and play back on meshcat
    meshcat.StartRecording()
    st = time.time()
    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(1.0)
    simulator.AdvanceTo(20.0)
    wall_time = time.time() - st
    print(f"sim time: {simulator.get_context().get_time():.4f}, "
           f"wall time: {wall_time:.4f}")
    meshcat.StopRecording()
    meshcat.PublishRecording()

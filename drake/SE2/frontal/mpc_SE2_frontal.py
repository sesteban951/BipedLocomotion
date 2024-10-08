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
    MultibodyPlant,
    VectorLogSink
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

from trajectory_generator_SE2_frontal import HLIPTrajectoryGeneratorSE2

import sys, os
grandparent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(grandparent_dir)
from mpc_utils import Interpolator, ModelPredictiveController
from joystick import GamepadCommand

#--------------------------------------------------------------------------------------------------------------------------#

def standing_position():
    """
    Return a reasonable default standing position for the Achilles humanoid. 
    """

    # no arms (parallel legs, bent knees)
    q_stand = np.array([          # (parallel legs, bent knees)
            0.0000, 0.9400,             # base position
            0.0000,                     # base orientation
            0, -0.5515, 1.0239,-0.4725, # left leg
            0, -0.5515, 1.0239,-0.4725, # right leg
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
    problem.num_steps = 20
    problem.q_init = np.copy(q_stand)
    problem.v_init = np.zeros(nv)
    
    # no arms
    problem.Qq = np.diag([
            15.0, 10.0,          # base position
            10.0,                # base orientation
            9.0, 7.0, 7.0, 5.0,  # left leg
            9.0, 7.0, 7.0, 5.0   # right leg
    ])
    problem.Qv = np.diag([
            15.0, 1.0,           # base position
            10.0,                # base orientation
            1.0, 1.0, 1.0, 1.0,  # left leg
            1.0, 1.0, 1.0, 1.0   # right leg
    ])
    problem.R = np.diag([
        100.0, 100.0,               # base position
        100.0,                      # base orientation
        0.0001, 0.0001, 0.0001, 0.0001,         # left leg
        0.0001, 0.0001, 0.0001, 0.0001          # right leg
    ])
    problem.Qf_q = 2.0 * np.copy(problem.Qq)
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

class AchillesPlanarMPC(ModelPredictiveController):
    """
    A Model Predictive Controller for the Achilles humanoid.
    """
    def __init__(self, optimizer, q_guess, mpc_rate, model_file):

        # inherit from the ModelPredictiveController class
        ModelPredictiveController.__init__(self, optimizer, q_guess, 11, 11, mpc_rate)

        self.joystick_port = self.DeclareVectorInputPort("joy_command",
                                                         BasicVector(5))  # LS_x, LS_y, RS_x, A button, RT (Xbox)

        # create internal model of the robot, TODO: there has to be a better way to do this
        self.plant = MultibodyPlant(0)
        Parser(self.plant).AddModels(model_file)
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        # time parameters
        self.t_current = 0.0       # current sim time
        self.t_phase = 0.0         # current phase time
        self.T_SSP = 0.3           # swing phase duration
        self.number_of_steps = -1   # number of individual swing foot steps taken

        #  z height parameters
        z_com_nom = 0.62    # nominal CoM height
        bezier_order = 7   # 5 or 7
        z_apex = 0.06      # apex height
        z_foot_offset = 0.01  # foot offset from the ground

        # maximum velocity for the robot
        self.v_max = 0.2

        # foot info variables
        self.left_foot_frame = self.plant.GetFrameByName("left_foot")
        self.right_foot_frame = self.plant.GetFrameByName("right_foot")
        self.stance_foot_frame = self.right_foot_frame
        self.swing_foot_frame  = self.left_foot_frame
        self.p_stance = None
        self.p_swing_init = None

        # create an HLIP trajectory generator object and set the parameters
        self.traj_gen_HLIP = HLIPTrajectoryGeneratorSE2(model_file)
        self.traj_gen_HLIP.set_parameters(z_nom = z_com_nom,
                                          z_apex = z_apex,
                                          z_offset = z_foot_offset,
                                          bezier_order = bezier_order,
                                          T_SSP = self.T_SSP,
                                          dt = self.optimizer.time_step(),
                                          N = self.optimizer.num_steps() + 1)

        # nominal standing configuration for MPC
        self.q_stand = standing_position()

        # computing the alpha value based on speed
        p = 0.8
        self.alpha = lambda v: ((1/self.v_max) * abs(v)) ** p

    # update the foot info for the HLIP traj gen
    def UpdateFootInfo(self):

        # check if entered new step period
        if (self.t_phase >= self.T_SSP) or (self.p_stance is None):
            
            # left foot is swing foot, right foot is stance foot
            if self.number_of_steps % 2 == 0:

                self.p_stance = self.plant.CalcPointsPositions(self.plant_context,
                                                                self.right_foot_frame,
                                                                [0,0,0],
                                                                self.plant.world_frame())
                self.p_swing_init = self.plant.CalcPointsPositions(self.plant_context,
                                                                    self.left_foot_frame,
                                                                    [0,0,0],
                                                                    self.plant.world_frame())

                self.stance_foot_frame = self.right_foot_frame
                self.swing_foot_frme = self.left_foot_frame

            # right foot is swing foot, left foot is stance foot
            else:

                self.p_stance = self.plant.CalcPointsPositions(self.plant_context,
                                                               self.left_foot_frame,
                                                               [0,0,0],
                                                               self.plant.world_frame())
                self.p_swing_init = self.plant.CalcPointsPositions(self.plant_context,
                                                                   self.right_foot_frame,
                                                                   [0,0,0],
                                                                   self.plant.world_frame())

                self.stance_foot_frame = self.left_foot_frame
                self.swing_foot_frame = self.right_foot_frame

            self.number_of_steps += 1

        # update the phase time
        self.t_phase = self.t_current - self.number_of_steps * self.T_SSP

    def UpdateNominalTrajectory(self, context):
        """
        Shift the reference trajectory based on the current position.
        """
        # Get the current state
        x0 = self.state_input_port.Eval(context)
        self.plant.SetPositionsAndVelocities(self.plant_context, x0)
        q0 = x0[:self.nq]
        v0 = x0[self.nq:]

        # Quit if the robot has fallen down
        base_height = q0[1]
        assert base_height > 0.2, "Oh no, the robot fell over!"

        # Get the current time
        self.t_current = context.get_time()

        print("-" * 75)
        print(f"t_current: {self.t_current:.3f}, t_phase: {self.t_phase:.3f}")

        # unpack the joystick commands
        joy_command = self.joystick_port.Eval(context)
        vy_des = joy_command[0] * self.v_max

        # Get the desired MPC standing trajectory
        q_stand = [np.copy(self.q_stand) for i in range(self.optimizer.num_steps() + 1)]
        v_stand = [np.copy(np.zeros(len(v0))) for i in range(self.optimizer.num_steps() + 1)]
        for i in range(self.num_steps + 1):
            q_stand[i][0] = q0[0] + vy_des * i * self.optimizer.time_step()
            v_stand[i][0] = vy_des

        # update the foot info 
        self.UpdateFootInfo()

        # get a new reference trajectory
        q_HLIP, v_HLIP = self.traj_gen_HLIP.generate_trajectory(q0 = q0,
                                                                v0 = v0,
                                                                v_des = vy_des,
                                                                t_phase = self.t_phase,
                                                                initial_swing_foot_pos = self.p_swing_init,
                                                                stance_foot_pos = self.p_stance,
                                                                initial_stance_foot_name = self.stance_foot_frame.name())
        for i in range(self.num_steps + 1):
            q_HLIP[i][0] = q0[0] + vy_des * i * self.optimizer.time_step()
            v_HLIP[i][0] = vy_des

        # compute alpha
        a = self.alpha(vy_des)

        # convex combination of the standing position and the nominal trajectory
        q_nom = [np.copy(np.zeros(len(q0))) for i in range(self.optimizer.num_steps() + 1)]
        v_nom = [np.copy(np.zeros(len(v0))) for i in range(self.optimizer.num_steps() + 1)]
        for i in range(self.optimizer.num_steps() + 1):
            q_nom[i] = (1 - a) * q_stand[i] + a * q_HLIP[i]
            v_nom[i] = (1 - a) * v_stand[i] + a * v_HLIP[i]

        self.optimizer.UpdateNominalTrajectory(q_nom, v_nom)

############################################################################################################################

if __name__=="__main__":

    meshcat = StartMeshcat()
    model_file = "../../../models/achilles_SE2_drake_frontal.urdf"

    # Set up a Drake diagram for simulation
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=5e-3)
    plant.set_discrete_contact_approximation(DiscreteContactApproximation.kLagged)

    models = Parser(plant).AddModels(model_file)  # robot model
    
    plant.RegisterCollisionGeometry(  # ground
        plant.world_body(), 
        RigidTransform(p=[0, 0, -25]), 
        Box(50, 50, 50), "ground", 
        CoulombFriction(0.7, 0.7))

    # Add implicit PD controllers (must use kLagged or kSimilar)
    kp_hip = 750
    kp_knee = 750
    kp_ankle = 150
    kd_hip = 10
    kd_knee = 10
    kd_ankle = 1
    
    # no arms
    Kp = np.array([kp_hip, kp_hip, kp_knee, kp_ankle, kp_hip, kp_hip, kp_knee, kp_ankle])
    Kd = np.array([kd_hip, kd_hip, kd_knee, kd_ankle, kd_hip, kd_hip, kd_knee, kd_ankle])
    
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
    controller = builder.AddSystem(AchillesPlanarMPC(optimizer, q_guess, mpc_rate, model_file))

    Bv = plant.MakeActuationMatrix()
    N = plant.MakeVelocityToQDotMap(plant.CreateDefaultContext())
    Bq = N@Bv
    interpolator = builder.AddSystem(Interpolator(Bq.T, Bv.T))

    # Logger to record the robot state
    logger = builder.AddSystem(VectorLogSink(plant.num_positions() + plant.num_velocities()))
    builder.Connect(
            plant.get_state_output_port(), 
            logger.get_input_port())

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
    simulator.AdvanceTo(15.0)
    wall_time = time.time() - st
    print(f"sim time: {simulator.get_context().get_time():.4f}, "
           f"wall time: {wall_time:.4f}")
    meshcat.StopRecording()
    meshcat.PublishRecording()

    # unpack recorded data from the logger
    log = logger.FindLog(diagram_context)
    times = log.sample_times()
    states = log.data().T

    # save the data to a CSV file
    with open('./data/data_SE2_frontal.csv', mode='w') as file:
        writer = csv.writer(file)
        for i in range(len(times)):
            writer.writerow([times[i]] + list(states[i]))

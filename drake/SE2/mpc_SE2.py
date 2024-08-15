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
    BasicVector
)

import time
import numpy as np

from pyidto import (
    TrajectoryOptimizer,
    TrajectoryOptimizerSolution,
    TrajectoryOptimizerStats,
    SolverParameters,
    ProblemDefinition,
    FindIdtoResource,
)

from trajectory_generator_SE2 import HLIPTrajectoryGeneratorSE2

import sys, os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from mpc_utils import Interpolator, ModelPredictiveController
from joystick import GamepadCommand

def standing_position():
    """
    Return a reasonable default standing position for the Achilles humanoid. 
    """
    # no arms (split legs, bent knees)
    # q_stand = np.array([
    #     0.0000, 0.9300,          # base position
    #     0.0000,                  # base orientation
    #    -0.5515, 1.0239,-0.4725,  # left leg
    #    -0.3200, 0.9751,-0.6552,  # right leg
    # ])
    # no arms (parallel legs, bent knees)
    q_stand = np.array([
        0.0000, 0.9300,          # base position
        0.0000,                  # base orientation
       -0.5515, 1.0239,-0.4725,  # left leg
       -0.5515, 1.0239,-0.4725,  # left leg
    ])

    # with arms (parallel legs, bent knees)
    # q_stand = np.array([
    #     0.0000, 0.930,          # base position
    #     0.0000,                  # base orientation
    #    -0.5515, 1.0239,-0.4725,  # left leg
    #    -0.0000, -0.0000,         # left arm
    #    -0.5515, 1.0239,-0.4725,  # right leg
    #    -0.0000, -0.0000,         # right arm
    # ])

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
        CoulombFriction(0.5, 0.5))
    plant.Finalize()
    diagram = builder.Build()

    nq = plant.num_positions()
    nv = plant.num_velocities()

    q_stand = standing_position()

    # Specify a cost function and target trajectory
    problem = ProblemDefinition()
    problem.num_steps = 15
    problem.q_init = np.copy(q_stand)
    problem.v_init = np.zeros(nv)
    
    # no arms
    problem.Qq = np.diag([
            35.0, 35.0,          # base position
            35.0,                # base orientation
            35.0, 35.0, 35.0,     # left leg
            35.0, 35.0, 35.0,     # right leg
    ])
    problem.Qv = np.diag([
            10.0, 10.0,            # base position
            10.0,                 # base orientation
            5.0, 5.0, 5.0,       # left leg
            5.0, 5.0, 5.0,       # right leg
    ])
    problem.R = 0.01 * np.diag([
        200.0, 200.0,               # base position
        200.0,                      # base orientation
        0.5, 0.5, 0.5,              # left leg
        0.5, 0.5, 0.5,              # right leg
    ])

    # with arms
    # problem.Qq = np.diag([
    #         35.0, 5.0,           # base position
    #         25.0,                # base orientation
    #         10.0, 10.0, 1.0,     # left leg
    #         10.0, 10.0,          # left arm
    #         10.0, 10.0, 1.0,     # right leg
    #         10.0, 10.0,          # right arm
    # ])
    # problem.Qv = np.diag([
    #         5.0, 0.2,            # base position
    #         0.2,                 # base orientation
    #         0.2, 0.2, 0.01,       # left leg
    #         0.01, 0.01,          # left arm
    #         0.2, 0.2, 0.01,       # right leg
    #         0.01, 0.01,          # right arm
    # ])
    # problem.R = 0.01 * np.diag([
    #     200.0, 200.0,               # base position
    #     200.0,                      # base orientation
    #     0.1, 0.1, 0.1,              # left leg
    #     0.1, 0.1,                   # left arm
    #     0.1, 0.1, 0.1,              # right leg
    #     0.1, 0.1,                   # right arm
    # ])

    problem.Qf_q = 1.0 * np.copy(problem.Qq)
    problem.Qf_v = 1.0 * np.copy(problem.Qv)

    v_nom = np.zeros(nv)
    problem.q_nom = [np.copy(q_stand) for i in range(problem.num_steps + 1)]
    problem.v_nom = [np.copy(v_nom) for i in range(problem.num_steps + 1)]

    # Set the solver parameters
    params = SolverParameters()
    params.max_iterations = 2
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
    params.verbose = True

    # Create the optimizer
    optimizer = TrajectoryOptimizer(diagram, plant, problem, params)

    # Return the optimizer, along with the diangram and plant, which must
    # stay in scope along with the optimizer
    return optimizer, diagram, plant


class AchillesPlanarMPC(ModelPredictiveController):
    """
    A Model Predictive Controller for the Achilles humanoid.
    """
    def __init__(self, optimizer, q_guess, mpc_rate, model_file):
        nq = q_guess[0].shape[0]
        ModelPredictiveController.__init__(self, optimizer, q_guess, nq, nq, mpc_rate)

        self.gamepad_port = self.DeclareVectorInputPort("joy_command",
                                                        BasicVector(4))  # LS_x, LS_y, RS_x, A button (Xbox)

        # create an HLIP trajectory generator object
        self.traj_gen_HLIP = HLIPTrajectoryGeneratorSE2(model_file)
        self.traj_gen_HLIP.z_apex = 0.05
        self.traj_gen_HLIP.T_SSP = 0.3
        
        self.foot_stance = "left_foot"
        self.S2S_steps = 0
        self.t_phase = 0.0
        self.vx_des = 0.0

        # indices to replace in the whole body trajectory using the IK trajectory
        # self.wb_idx = [0,1,2,3,4,5,8,9,10]
        self.wb_idx = [3,4,5,6,7,8]
        self.ik_idx = [3,4,5,6,7,8]

    def UpdateNominalTrajectory(self, context):
        """
        Shift the reference trajectory based on the current position.
        """
        # Get the current state
        x0 = self.state_input_port.Eval(context)
        q0 = x0[:self.nq]
        v0 = x0[self.nq:]

        # Get the current time
        t_current = context.get_time()

        # unpack the joystick commands

        # Quit if the robot has fallen down
        base_height = q0[1]
        assert base_height > 0.3, "Oh no, the robot fell over!"

        # Get the current nominal trajectory
        prob = self.optimizer.prob()
        q_nom = prob.q_nom
        v_nom = prob.v_nom

        # Shift the nominal trajectory
        dt = self.optimizer.time_step()
        for i in range(self.num_steps + 1):
            q_nom[i][0] = q0[0] + self.vx_des * i * dt
            v_nom[i][0] = self.vx_des

        # # check stance foot
        # self.t_phase = t_current - self.traj_gen_HLIP.T_SSP * self.S2S_steps
        # if (t_current >= self.traj_gen_HLIP.T_SSP * (self.S2S_steps + 1)):
        #     if self.foot_stance == "left_foot":
        #         self.foot_stance = "right_foot"
        #     else:
        #         self.foot_stance = "left_foot"
        #     self.t_phase = t_current
        #     self.S2S_steps += 1

        # print("foot_stance: ", self.foot_stance)
        # print("S2S_steps: ", self.S2S_steps)
        # print("t_phase: ", self.t_phase)

        # # update the trajectory based on HLIP 
        # q_legs, v_legs = self.traj_gen_HLIP.get_trajectory(q0 = np.array(q0),
        #                                                    v0 = np.array(v0),
        #                                                    initial_stance_foot = self.foot_stance,
        #                                                    v_des = self.vx_des,
        #                                                    z_nom = 0.65,
        #                                                    dt = self.time_step,
        #                                                    N = self.num_steps + 1)

        # for i in range(len(q_legs)):
        #     q_ik = q_legs[i]
        #     v_ik = v_legs[i]
        #     for wb_idx, ik_idx in zip(self.wb_idx, self.ik_idx):
        #         q_nom[i][wb_idx] = q_ik[ik_idx]
        #         v_nom[i][wb_idx] = v_ik[ik_idx]

        # for i in range(self.num_steps + 1):
        #     q_nom[i][0] = q0[0] + self.vx_des * i * self.time_step
        #     v_nom[i][0] = self.vx_des
        #     q_nom[i][2] = 0
        #     v_nom[i][2] = 0

        self.optimizer.UpdateNominalTrajectory(q_nom, v_nom)

if __name__=="__main__":

    meshcat = StartMeshcat()
    model_file = "../../models/achilles_SE2_drake.urdf"
    # model_file = "../../models/achilles_SE2_arms_drake.urdf"

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
    kp_arm = 50
    kd_hip = 10
    kd_knee = 10
    kd_ankle = 1
    kd_arm = 1
    
    # no arms
    Kp = np.array([kp_hip, kp_knee, kp_ankle, kp_hip, kp_knee, kp_ankle])
    Kd = np.array([kd_hip, kd_knee, kd_ankle, kd_hip, kd_knee, kd_ankle])
    
    # with arms
    # Kp = np.array([kp_hip, kp_knee, kp_ankle, kp_arm, kp_arm, kp_hip, kp_knee, kp_ankle, kp_arm, kp_arm])
    # Kd = np.array([kd_hip, kd_knee, kd_ankle, kd_arm, kd_arm, kd_hip, kd_knee, kd_ankle, kd_arm, kd_arm])
    
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
    mpc_rate = 75  # Hz
    controller = builder.AddSystem(AchillesPlanarMPC(optimizer, q_guess, mpc_rate, model_file))

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
    simulator.AdvanceTo(10.0)
    wall_time = time.time() - st
    print(f"sim time: {simulator.get_context().get_time():.4f}, "
           f"wall time: {wall_time:.4f}")
    meshcat.StopRecording()
    meshcat.PublishRecording()

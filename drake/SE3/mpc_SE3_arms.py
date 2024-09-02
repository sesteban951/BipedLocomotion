#!/usr/bin/env python

##
#
# MPC with 3D+arms version of the Achilles humanoid.
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
    VectorLogSink,
    RollPitchYaw,
    RotationMatrix,
    Rgba, Sphere
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

from trajectory_generator_SE3 import HLIPTrajectoryGeneratorSE3

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

    # no arms (parallel legs, bent knees)
    q_stand = np.array([
        1.0000, 0.0000, 0.0000, 0.0000,            # base orientation, (w, x, y, z)
        0.0000, 0.0000, 0.9300,                    # base position, (x,y,z)
        0.0000, 0.0209, -0.5515, 1.0239,-0.4725,   # left leg, (hip yaw, hip roll, hip pitch, knee, ankle) 
        0.0000, 0.0000, 0.0000, 0.0000,            # left arm, (shoulder pitch, shoulder roll, shoulder yaw, elbow)
        0.0000, -0.0209, -0.5515, 1.0239,-0.4725,  # right leg, (hip yaw, hip roll, hip pitch, knee, ankle) 
        0.0000, 0.0000, 0.0000, 0.0000             # right arm, (shoulder pitch, shoulder roll, shoulder yaw, elbow)
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
        1.0, 1.0, 1.0, 1.0,      # base orientation
        1.0, 1.0, 1.0,           # base position
        0.1, 1.0, 0.1, 0.1, 0.1, # left leg
        0.05, 0.05, 0.05, 0.05,  # left arm
        0.1, 1.0, 0.1, 0.1, 0.1, # left leg
        0.05, 0.05, 0.05, 0.05   # left arm
    ])
    problem.Qv = np.diag([
        0.1, 0.1, 0.1,               # base orientation
        0.1, 0.1, 0.1,               # base position
        0.01, 0.1, 0.01, 0.01, 0.01, # left leg
        0.01, 0.01, 0.01, 0.01,      # left arm
        0.01, 0.1, 0.01, 0.01, 0.01, # right leg
        0.01, 0.01, 0.01, 0.01       # right arm
    ])
    problem.R = np.diag([
        50.0, 50.0, 50.0,                       # base orientation
        50.0, 50.0, 50.0,                       # base position
        0.0001, 0.0001, 0.0001, 0.0001, 0.0001, # left leg
        0.0001, 0.0001, 0.0001, 0.0001,         # left arm
        0.0001, 0.0001, 0.0001, 0.0001, 0.0001, # right leg
        0.0001, 0.0001, 0.0001, 0.0001          # right arm
    ])
    problem.Qf_q = 10.0 * np.copy(problem.Qq)
    problem.Qf_v = 10.0 * np.copy(problem.Qv)

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
    params.contact_stiffness = 5_000
    params.dissipation_velocity = 0.1
    params.smoothing_factor = 0.01
    params.friction_coefficient = 0.9
    params.stiction_velocity = 0.5
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
    def __init__(self, optimizer, q_guess, mpc_rate, model_file, meshcat):

        # inherit from the ModelPredictiveController class
        ModelPredictiveController.__init__(self, optimizer, q_guess, 25, 24, mpc_rate)

        # make a copy of meshcat
        self.meshcat = meshcat

        # create input port for the joystick command
        self.joystick_port = self.DeclareVectorInputPort("joy_command",
                                                         BasicVector(5))  # LS_x, LS_y, RS_x, A button, RT (Xbox)
        
        # create internal model of the robot, TODO: there has to be a better way to do this!!!!!!!!!!!
        self.plant = MultibodyPlant(0)
        Parser(self.plant).AddModels(model_file)
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        # time parameters
        self.t_current = 0.0       # current sim time
        self.t_phase = 0.0         # current phase time
        self.T_SSP = 0.3           # swing phase duration
        self.number_of_steps = -1   # number of individual swing foot steps taken

        # z height parameters
        self.z_com_upper = 0.64   # upper CoM height
        self.z_com_lower = 0.45   # lower CoM height
        z_apex = 0.06           # apex height
        z_foot_offset = 0.01    # foot offset from the ground
        bezier_order = 7        # 5 or 7
        hip_bias = 0.3          # bias between the foot-to-foot distance in y-direction

        # maximum velocity for the robot
        self.v_max = 0.2  # [m/s]
        self.w_max = 25   # [deg/s]

        # foot info variables
        self.left_foot_frame = self.plant.GetFrameByName("left_foot")
        self.right_foot_frame = self.plant.GetFrameByName("right_foot")
        self.stance_foot_frame = self.right_foot_frame
        self.swing_foot_frame  = self.left_foot_frame
        self.p_stance_W = None
        self.R_stance_W = None
        self.R_stance_W_2D = None
        self.p_swing_init_W = None
        self.quat_stance = None
        self.control_stance_yaw = None

        # get the constant offset of the torso frame in the CoM frame
        torso_frame = self.plant.GetFrameByName("torso")
        static_com_frame = self.plant.GetFrameByName("static_com")
        self.p_torso_com = self.plant.CalcPointsPositions(self.plant_context,
                                                          torso_frame,
                                                          [0, 0, 0],
                                                          static_com_frame)

        # create an HLIP trajectory generator object and set the parameters
        self.traj_gen_HLIP = HLIPTrajectoryGeneratorSE3(model_file)
        self.traj_gen_HLIP.set_parameters(z_apex = z_apex,
                                          z_offset = z_foot_offset,
                                          hip_bias = hip_bias,
                                          bezier_order = bezier_order,
                                          T_SSP = self.T_SSP,
                                          dt = self.optimizer.time_step(),
                                          N = self.optimizer.num_steps() + 1)
        
        # nominal standing configuration for MPC
        self.q_stand = standing_position()

        # computing the alpha value based on speed
        p = 2.0
        self.alpha = lambda v_des: ((1/self.v_max) * v_des) ** p

        # index of whole body that are leg joints
        self.idx_no_arms_q = [0, 1, 2, 3,         # quat coords
                              4, 5, 6,            # pos coords
                              7, 8, 9, 10, 11,    # left leg
                              16, 17, 18, 19, 20] # right leg
        self.idx_no_arms_v = [0, 1, 2,            # omega coords
                              3, 4, 5,            # v coords
                              6, 7, 8, 9, 10,     # left leg
                              15, 16, 17, 18, 19] # right leg

        # create someobjects for the meshcat plot
        red_color = Rgba(1, 0, 0, 1)
        green_color = Rgba(0, 1, 0, 1)
        blue_color = Rgba(0, 0, 1, 1)
        sphere_com = Sphere(0.018)
        sphere_foot = Sphere(0.015)

        # create separate objects for the CoM and the feet visualization
        for i in range(self.optimizer.num_steps() + 1):
            self.meshcat.SetObject(f"com_{i}", sphere_com, green_color)
            self.meshcat.SetObject(f"left_{i}", sphere_foot, blue_color)
            self.meshcat.SetObject(f"right_{i}", sphere_foot, red_color)

    #----------------------------------------------------------------------------------------------------------------------#

    # update the meshcat plot with the HLIP horizon
    def plot_meshcat_horizon(self, meshcat_horizon, t):

        for i in range(self.optimizer.num_steps() + 1):
            
            # unpack the horizon data
            O = meshcat_horizon[i]
            p_com, p_left, p_right = O

            # plot the foot and com positions
            self.meshcat.SetTransform(f"com_{i}", RigidTransform(p_com), t)
            self.meshcat.SetTransform(f"left_{i}", RigidTransform(p_left), t)
            self.meshcat.SetTransform(f"right_{i}", RigidTransform(p_right), t)

    # increment a current quaternion given omega_z
    def increment_quaternion(self, quat_0, omega_z, T):

        # convert the quaternion to a RollPitchYaw object
        rpy = RollPitchYaw([0, 0, omega_z * T])
        quat_delta = rpy.ToQuaternion()

        # multiply the quaternions
        quat_final = quat_delta.multiply(quat_0)

        # return the final incremented quaternion
        return quat_final

    # update the foot info for the HLIP traj gen
    def UpdateFootInfo(self):

        # check if entered new step period
        if (self.t_phase >= self.T_SSP) or (self.p_stance_W is None):

            # increment the number of steps
            self.number_of_steps += 1

            # left foot is swing foot, right foot is stance foot
            if self.number_of_steps % 2 == 0:

                # set the initial swing foot position
                self.p_swing_init_W = self.plant.CalcPointsPositions(self.plant_context,
                                                                    self.left_foot_frame,
                                                                    [0,0,0],
                                                                    self.plant.world_frame())
                # set the current stance foot position
                self.p_stance_W = self.plant.CalcPointsPositions(self.plant_context,
                                                                self.right_foot_frame,
                                                                [0,0,0],
                                                                self.plant.world_frame())
                # get the current stance yaw position of the robot
                self.R_stance_W = self.plant.CalcRelativeRotationMatrix(self.plant_context,
                                                                        self.plant.world_frame(),
                                                                        self.right_foot_frame)
                # switch the foot roles
                self.stance_foot_frame = self.right_foot_frame
                self.swing_foot_frame = self.left_foot_frame

            # right foot is swing foot, left foot is stance foot
            else:

                # set the initial swing foot position
                self.p_swing_init_W = self.plant.CalcPointsPositions(self.plant_context,
                                                                     self.right_foot_frame,
                                                                     [0,0,0],
                                                                     self.plant.world_frame())
                # set the current stance foot position
                self.p_stance_W = self.plant.CalcPointsPositions(self.plant_context,
                                                                 self.left_foot_frame,
                                                                 [0,0,0],
                                                                 self.plant.world_frame())
                # get the current stance yaw position of the robot
                self.R_stance_W = self.plant.CalcRelativeRotationMatrix(self.plant_context,
                                                                        self.plant.world_frame(),
                                                                        self.left_foot_frame)
                # switch the foot roles
                self.stance_foot_frame = self.left_foot_frame
                self.swing_foot_frame = self.right_foot_frame
            
            # compute the stance foot yaw angle and the 2D rotation matrix
            self.control_stance_yaw = RollPitchYaw(self.R_stance_W).yaw_angle()  # NOTE: could cause singularity
            self.quat_stance = RollPitchYaw([0,0,self.control_stance_yaw]).ToQuaternion()
            self.R_satnce_W = RotationMatrix(self.quat_stance).matrix()
            self.R_stance_W_2D = np.array([[np.cos(self.control_stance_yaw), -np.sin(self.control_stance_yaw)],
                                           [np.sin(self.control_stance_yaw),  np.cos(self.control_stance_yaw)]])

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

        # Update the foot info
        self.UpdateFootInfo()

        # Quit if the robot has fallen down
        base_height = q0[6]
        assert base_height > 0.4, "Oh no, the robot fell over!"

        # Get the current time
        self.t_current = context.get_time()

        # unpack the joystick commands
        joy_command = self.joystick_port.Eval(context)
        vx_des = joy_command[1] * self.v_max  
        vy_des = joy_command[0] * self.v_max
        wz_des = joy_command[2] * self.w_max * (np.pi / 180)
        z_com_des = joy_command[4] * (self.z_com_lower - self.z_com_upper) + self.z_com_upper
        v_des = np.array([[vx_des], [vy_des]]) # in local stance foot frame

        # Get the desired MPC standing trajectory
        q_stand = [np.copy(self.q_stand) for i in range(self.optimizer.num_steps() + 1)]
        v_stand = [np.copy(np.zeros(len(v0))) for i in range(self.optimizer.num_steps() + 1)]
        v_des_W = self.R_stance_W_2D @ v_des
        for i in range(self.num_steps + 1):

            # increment orientation
            quat_delta = self.increment_quaternion(self.quat_stance, wz_des, i * self.optimizer.time_step())
            q_stand[i][0] = quat_delta.w()
            q_stand[i][1] = quat_delta.x()
            q_stand[i][2] = quat_delta.y()
            q_stand[i][3] = quat_delta.z()
            
            # increment position
            p_increment = self.R_stance_W_2D @ (v_des * i * self.optimizer.time_step())
            q_stand[i][4] = q0[4] + p_increment[0][0]
            q_stand[i][5] = q0[5] + p_increment[1][0]
            q_stand[i][6] = z_com_des + self.p_torso_com[2][0]
            v_stand[i][3] = v_des_W[0][0]
            v_stand[i][4] = v_des_W[1][0]

        print("------------------------------------------------------------")
        print(f"t_current: {self.t_current}")

        # get a new reference trajectory
        q_HLIP, v_HLIP, meshcat_horizon = self.traj_gen_HLIP.generate_trajectory(
                                                                q0 = q0,
                                                                v0 = v0,
                                                                v_des = v_des,
                                                                z_com_des = z_com_des,
                                                                t_phase = self.t_phase,
                                                                initial_swing_foot_pos = self.p_swing_init_W,
                                                                stance_foot_pos = self.p_stance_W,
                                                                stance_foot_yaw = self.control_stance_yaw,
                                                                initial_stance_foot_name = self.stance_foot_frame.name())
        # replace HLIP floating base reference trajectory
        for i in range(self.num_steps + 1):
            
            # increment orientation
            quat_delta = self.increment_quaternion(self.quat_stance, wz_des, i * self.optimizer.time_step())
            q_HLIP[i][0] = quat_delta.w()
            q_HLIP[i][1] = quat_delta.x()
            q_HLIP[i][2] = quat_delta.y()
            q_HLIP[i][3] = quat_delta.z()
            
            # increment position
            p_increment = self.R_stance_W_2D @ (v_des * i * self.optimizer.time_step())
            q_HLIP[i][4] = q0[4] + p_increment[0][0]
            q_HLIP[i][5] = q0[5] + p_increment[1][0]
            q_stand[i][6] = z_com_des + self.p_torso_com[2][0]
            v_HLIP[i][3] = v_des_W[0][0]
            v_HLIP[i][4] = v_des_W[1][0]

        # draw the meshcat horizon
        self.plot_meshcat_horizon(meshcat_horizon, self.t_current)

        # compute alpha 
        # v_norm = np.linalg.norm([vx_des, vy_des, wz_des])
        # a = self.alpha(v_norm)

        # # clamp a to [0, 1]
        # a = np.clip(a, 0, 1) # TODO: alpha is not mapping joystick vel to [0,1], do this more intelligently
        # print(f"alpha: {a}")

        # # convex combination of the standing position and the nominal trajectory
        # q_nom = [np.copy(np.zeros(len(q0))) for i in range(self.optimizer.num_steps() + 1)]
        # v_nom = [np.copy(np.zeros(len(v0))) for i in range(self.optimizer.num_steps() + 1)]
        # for i in range(self.optimizer.num_steps() + 1):
        #     q_nom[i][self.idx_no_arms_q] = (1 - a) * q_stand[i][self.idx_no_arms_q] + a * q_HLIP[i]
        #     v_nom[i][self.idx_no_arms_v] = (1 - a) * v_stand[i][self.idx_no_arms_v] + a * v_HLIP[i]

        q_nom = q_stand
        v_nom = v_stand
        # q_nom = q_HLIP
        # v_nom = v_HLIP

        self.optimizer.UpdateNominalTrajectory(q_nom, v_nom)

############################################################################################################################

if __name__=="__main__":

    meshcat = StartMeshcat()
    model_file = "../../models/achilles_drake.urdf"

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
    kp_hip = 900
    kp_knee = 1000
    kp_ankle = 150
    kp_arm = 100
    kd_hip = 10
    kd_knee = 10
    kd_ankle = 1
    kd_arm = 1
    Kp = np.array([kp_hip, kp_hip, kp_hip, kp_knee, kp_ankle, 
                   kp_arm, kp_arm, kp_arm, kp_arm,
                   kp_hip, kp_hip, kp_hip, kp_knee, kp_ankle,
                   kp_arm, kp_arm, kp_arm, kp_arm])
    Kd = np.array([kd_hip, kd_hip, kd_hip, kd_knee, kd_ankle, 
                   kd_arm, kd_arm, kd_arm, kd_arm,
                   kd_hip, kd_hip, kd_hip, kd_knee, kd_ankle,
                   kd_arm, kd_arm, kd_arm, kd_arm])
    
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
    controller = builder.AddSystem(AchillesMPC(optimizer, q_guess, mpc_rate, model_file, meshcat))

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
    with open('./data/data_SE3_arms.csv', mode='w') as file:
        writer = csv.writer(file)
        for i in range(len(times)):
            writer.writerow([times[i]] + list(states[i]))

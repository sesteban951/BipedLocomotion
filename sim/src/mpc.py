#!/usr/bin/env python

##
#
# MPC with 3D+arms version of the Achilles humanoid.
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
from disturbance_generator import DisturbanceGenerator        # type: ignore
from trajectory_generator import HLIPTrajectoryGenerator      # type: ignore
from contact_logger import ContactLogger                      # type: ignore

# import the yaml config
config_path = "../config/config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)

# Fix the random seed for reproducibility        
if config['seed']['fixed']:
    np.random.seed(config['seed']['value'])

#--------------------------------------------------------------------------------------------------------------------------#

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

    if config['wall']['enabled']==True:
        # Add a wall to the controller's model, if enabled
        R_wall = RigidTransform(
            p=[config['wall']['x'], config['wall']['y'], config['wall']['height']/2])
        wall_geom = Box(config['wall']['length'], 0.2, config['wall']['height'])
        plant.RegisterCollisionGeometry(
            plant.world_body(), 
            R_wall, wall_geom, "wall", CoulombFriction(0.7, 0.7))

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
        
        # create internal model of the robot
        self.plant = MultibodyPlant(0)
        Parser(self.plant).AddModels(model_file)
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        # time parameters
        self.t_current = 0.0                 # current sim time
        self.t_phase = 0.0                   # current phase time
        self.T_SSP = config['HLIP']['T_SSP'] # swing phase duration
        self.number_of_steps = -1            # number of individual swing foot steps taken

        # z height parameters
        self.z_com_upper = config['HLIP']['z_com_upper']  # upper CoM height
        self.z_com_lower = config['HLIP']['z_com_lower']  # lower CoM height
        z_apex = config['HLIP']['z_apex']                 # apex height
        z_foot_offset = config['HLIP']['z_foot_offset']   # foot offset from the ground
        bezier_order = config['HLIP']['bezier_order']     # 5 or 7
        hip_bias = config['HLIP']['hip_bias']             # bias between the foot-to-foot distance in y-direction

        # maximum velocity for the robot
        self.vx_max = config['HLIP']['vx_max']  # [m/s]
        self.vy_max = config['HLIP']['vy_max']  # [m/s]
        w_max = config['HLIP']['wz_max']        # [deg/s]
        self.w_max = w_max * (np.pi / 180)      # [rad/s]
        P_half = np.diag([1/self.vx_max, 1/self.vy_max, 1/self.w_max]) 
        self.P = P_half.T @ P_half

        # avtication function for blending
        a = config['blending']['a']  
        p = config['blending']['p']  
        tanh = lambda x: (np.exp(2*x) -1 ) / (np.exp(2*x) + 1)
        self.alpha = lambda v_des: 0.5 * tanh(a * (v_des - p)) + 0.5  

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
        self.traj_gen_HLIP = HLIPTrajectoryGenerator(model_file)
        self.traj_gen_HLIP.set_parameters(z_apex = z_apex,
                                          z_offset = z_foot_offset,
                                          hip_bias = hip_bias,
                                          bezier_order = bezier_order,
                                          T_SSP = self.T_SSP,
                                          dt = self.optimizer.time_step(),
                                          N = self.optimizer.num_steps() + 1)
        
        # nominal standing configuration for MPC
        self.q_stand = standing_position()

        # index of whole body that are leg joints
        self.idx_no_arms_q = [0, 1, 2, 3,         # quat coords
                              4, 5, 6,            # pos coords
                              7, 8, 9, 10, 11,    # left leg
                              16, 17, 18, 19, 20] # right leg
        self.idx_no_arms_v = [0, 1, 2,            # omega coords
                              3, 4, 5,            # v coords
                              6, 7, 8, 9, 10,     # left leg
                              15, 16, 17, 18, 19] # right leg

        if config['HLIP_vis']==True:
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

            # left foot is swing foot, fright foot is stance foot
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

        # Simulate state estimation error
        if config['estimation_error']['enabled']:
            x0 += np.random.normal(
                config['estimation_error']['mu'], 
                config['estimation_error']['sigma'], 
                len(x0))

        # Update the internal model
        self.plant.SetPositionsAndVelocities(self.plant_context, x0)
        q0 = x0[:self.nq]
        v0 = x0[self.nq:]

        # Update the foot info
        self.UpdateFootInfo()

        # Quit if the robot has fallen down
        base_height = q0[6]
        assert base_height > 0.1, "Oh no, the robot fell over!"

        # Get the current time
        self.t_current = context.get_time()

        # unpack the joystick commands
        if config['references']['enabled']==True:
            vx_des = config['references']['vx_ref']
            vy_des = config['references']['vy_ref']
            wz_des = config['references']['wz_ref'] * (np.pi / 180)
            z_com_des = config['references']['z_com_ref']
        else:
            joy_command = self.joystick_port.Eval(context)
            vx_des = joy_command[1] * self.vx_max  
            vy_des = joy_command[0] * self.vy_max
            wz_des = joy_command[2] * self.w_max
            z_com_des = joy_command[4] * (self.z_com_lower - self.z_com_upper) + self.z_com_upper
        v_des = np.array([[vx_des], [vy_des]]) # in local stance foot frame
        v_base = np.array([[v0[3]], [v0[4]], [v0[5]]]) # in world frame

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
        if config['HLIP_vis']==True:
            self.plot_meshcat_horizon(meshcat_horizon, self.t_current)

        # compute alpha 
        v_command = np.array([[vx_des], [vy_des], [wz_des]])
        v_command_norm = (v_command.T @ self.P @ v_command)[0][0] # NOTE: this is 1 if any of the axis sees its respective max velocity, otherwise can exceed 1.0 easily
        v_base_norm = np.linalg.norm(v_base)
        v = np.max([v_base_norm, v_command_norm])
        a = self.alpha(v)                                         # NOTE: activation function should be bounded [0,1]

        # convex combination of the standing position and the nominal trajectory
        q_nom = [np.copy(np.zeros(len(q0))) for i in range(self.optimizer.num_steps() + 1)]
        v_nom = [np.copy(np.zeros(len(v0))) for i in range(self.optimizer.num_steps() + 1)]
        for i in range(self.optimizer.num_steps() + 1):
            q_nom[i][self.idx_no_arms_q] = (1 - a) * q_stand[i][self.idx_no_arms_q] + a * q_HLIP[i]
            v_nom[i][self.idx_no_arms_v] = (1 - a) * v_stand[i][self.idx_no_arms_v] + a * v_HLIP[i]

        self.optimizer.UpdateNominalTrajectory(q_nom, v_nom)

#--------------------------------------------------------------------------------------------------------------------------#

class HLIP(LeafSystem):

    # constructor
    def __init__(self, model_file, meshcat):

        # init leaf system 
        LeafSystem.__init__(self)

        # meshcat
        self.meshcat = meshcat

        # create internal model of the robot
        self.plant = MultibodyPlant(0)
        Parser(self.plant).AddModels(model_file)
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        self.nq = self.plant.num_positions()
        self.nv = self.plant.num_velocities()

        # create input port for the joystick command
        self.state_input_port = self.DeclareVectorInputPort("state",
                                                            BasicVector(self.nq + self.nv))
        self.joystick_port = self.DeclareVectorInputPort("joy_command",
                                                         BasicVector(5))  # LS_x, LS_y, RS_x, A button, RT (Xbox)
        self.DeclareVectorOutputPort("x_des",
                                     BasicVector(2 * self.plant.num_actuators()),
                                     self.CalcOutput)    

        # time parameters
        self.t_current = 0.0                 # current sim time
        self.t_phase = 0.0                   # current phase time
        self.T_SSP = config['HLIP']['T_SSP'] # swing phase duration
        self.number_of_steps = -1            # number of individual swing foot steps taken

        # maximum velocity for the robot
        self.vx_max = config['HLIP']['vx_max']  # [m/s]
        self.vy_max = config['HLIP']['vy_max']  # [m/s]

        # z height parameters
        self.z_com_upper = config['HLIP']['z_com_upper']  # upper CoM height
        self.z_com_lower = config['HLIP']['z_com_lower']  # lower CoM height
        z_apex = config['HLIP']['z_apex']                 # apex height
        z_foot_offset = config['HLIP']['z_foot_offset']   # foot offset from the ground
        bezier_order = config['HLIP']['bezier_order']     # 5 or 7
        hip_bias = config['HLIP']['hip_bias']             # bias between the foot-to-foot distance in y-direction

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
        self.traj_gen_HLIP = HLIPTrajectoryGenerator(model_file)
        self.traj_gen_HLIP.set_parameters(z_apex = z_apex,
                                          z_offset = z_foot_offset,
                                          hip_bias = hip_bias,
                                          bezier_order = bezier_order,
                                          T_SSP = self.T_SSP,
                                          dt = 0.005,
                                          N = 2)

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

    def CalcOutput(self, context, output):

        # Get the current state
        x0 = self.state_input_port.Eval(context)
        
        # Simulate state estimation error
        if config['estimation_error']['enabled']:
            x0 += np.random.normal(
                config['estimation_error']['mu'], 
                config['estimation_error']['sigma'], 
                len(x0))
            
        # Update the internal model
        self.plant.SetPositionsAndVelocities(self.plant_context, x0)
        q0 = x0[:self.nq]
        v0 = x0[self.nq:]

        # Update the foot info
        self.UpdateFootInfo()

        # Quit if the robot has fallen down
        base_height = q0[6]
        assert base_height > 0.1, "Oh no, the robot fell over!"

        # Get the current time
        self.t_current = context.get_time()

        # get the desired velocity
        if config['references']['enabled']==True:
            vx_des = config['references']['vx_ref']
            vy_des = config['references']['vy_ref']
            z_com_des = config['references']['z_com_ref']
        # unpack the joystick commands
        else:
            joy_command = self.joystick_port.Eval(context)
            vx_des = joy_command[1] * self.vx_max  
            vy_des = joy_command[0] * self.vy_max
            z_com_des = joy_command[4] * (self.z_com_lower - self.z_com_upper) + self.z_com_upper
        v_des = np.array([[vx_des], [vy_des]]) # in local stance foot frame

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
        if config['HLIP_vis']==True:
            self.plot_meshcat_horizon(meshcat_horizon, self.t_current)
        
        # extract the first trajectory point
        q_des = q_HLIP[0]
        v_des = v_HLIP[0]

        # insert the arms
        q_des = np.array([q_des[7], q_des[8], q_des[9], q_des[10], q_des[11],
                          0.0, 0.0, 0.0, 0.0,
                          q_des[12], q_des[13], q_des[14], q_des[15], q_des[16],
                          0.0, 0.0, 0.0, 0.0])
        # v_des = np.array([v_des[6], v_des[7], v_des[8], v_des[9], v_des[10],
        #                   0.0, 0.0, 0.0, 0.0,
        #                   v_des[11], v_des[12], v_des[13], v_des[14], v_des[15],
        #                   0.0, 0.0, 0.0, 0.0])
        v_des = np.zeros(18)
        x_des = np.block([q_des, v_des])

        output.SetFromVector(x_des)

############################################################################################################################

if __name__=="__main__":

    # start meshcat
    meshcat = StartMeshcat()

    # set up the model file
    model_file = "../../models/achilles_drake.urdf"

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
   
    # Add rough terrain
    terrain_color = np.array(config['color']['terrain'])
    if config['terrain']['enabled']==True:
        for i in range(config['terrain']['num_entities']):
            
            px = np.random.uniform(config['terrain']['x_range'][0], config['terrain']['x_range'][1])
            py = np.random.uniform(config['terrain']['y_range'][0], config['terrain']['y_range'][1])
            radius = np.random.uniform(config['terrain']['r_range'][0], config['terrain']['r_range'][1])
            plant.RegisterVisualGeometry(
                plant.world_body(), 
                RigidTransform(p=[px, py, -0.1]), 
                Sphere(radius), f"terrain_{i}", 
                terrain_color)
            plant.RegisterCollisionGeometry(
                plant.world_body(), 
                RigidTransform(p=[px, py, -0.1]), 
                Sphere(radius), f"terrain_{i}", 
                CoulombFriction(0.7, 0.7))
            
    # Add a wall
    if config['wall']['enabled']==True:
        R_wall = RigidTransform(
            p=[config['wall']['x'], config['wall']['y'], config['wall']['height']/2])
        wall_geom = Box(config['wall']['length'], 0.2, config['wall']['height'])
        plant.RegisterVisualGeometry(
            plant.world_body(), 
            R_wall, 
            wall_geom, 
            "wall", 
            terrain_color)
        plant.RegisterCollisionGeometry(
            plant.world_body(), 
            R_wall, wall_geom, "wall", CoulombFriction(0.7, 0.7))
        
    # Add obstacles (visual only) to drive around
    if config["obstacle_field"]["enabled"]:
        for i in range(config["obstacle_field"]["num_obstacles"]):
            px = np.random.uniform(*config["obstacle_field"]["x_range"])
            py = np.random.uniform(*config["obstacle_field"]["y_range"])
            radius = np.random.uniform(*config["obstacle_field"]["radius_range"])
            
            plant.RegisterVisualGeometry(
                plant.world_body(), 
                RigidTransform(p=[px, py, 1.0]), 
                Cylinder(radius, 2.0), f"obs_{i}", 
                terrain_color)
            
    # Add an obstacle (visual only) to duck under
    if config["overhang"]["enabled"]:
        plant.RegisterVisualGeometry(
            plant.world_body(), 
            RigidTransform(p=config["overhang"]["position"]), 
            Box(*config["overhang"]["dimensions"]), "overhang", 
            terrain_color)

    # Add implicit PD controllers (must use kLagged or kSimilar)
    Kp = np.array(config['gains']['Kp'])
    Kd = np.array(config['gains']['Kd'])    
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
    if config['controller']=='MPC':
    
        mpc_rate = config['MPC']['mpc_rate']  # Hz
        controller = builder.AddSystem(AchillesMPC(optimizer, q_guess, mpc_rate, model_file, meshcat))

        Bv = plant.MakeActuationMatrix()
        N = plant.MakeVelocityToQDotMap(plant.CreateDefaultContext())
        Bq = N@Bv
        interpolator = builder.AddSystem(Interpolator(Bq.T, Bv.T))
    
    # Create the HLIP controller
    elif config['controller']=='HLIP':

        controller = builder.AddSystem(HLIP(model_file, meshcat))
    
    # distubance generator
    disturbance_tau = np.zeros(plant.num_velocities())
    d = np.random.multivariate_normal(
        config['disturbance']['mu'], 
        np.diag(config['disturbance']['sigma'])**2)
    disturbance_tau[3:6] = d
    time_applied = config['disturbance']['time_applied']
    duration = config['disturbance']['duration']
    dist_gen = builder.AddSystem(DisturbanceGenerator(plant, 
                                                      meshcat, 
                                                      disturbance_tau, time_applied, duration))
    print("Disturbance vector: ", d)

    # Wire the systems together
    if config['controller']=='MPC':
        
        # MPC
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
            plant.get_desired_state_input_port(models[0]))
        
    elif config['controller']=='HLIP':

        # HLIP
        builder.Connect(plant.get_state_output_port(), 
                controller.GetInputPort("state"))
        builder.Connect(controller.GetOutputPort("x_des"),
                plant.get_desired_state_input_port(models[0]))

    # joystick
    builder.Connect(
        joystick.get_output_port(), 
        controller.GetInputPort("joy_command"))
    
    # disturbances
    builder.Connect(
        dist_gen.get_output_port(),
        plant.get_applied_generalized_force_input_port())
    builder.Connect(
        plant.get_state_output_port(),
        dist_gen.GetInputPort("state"))
    
    # Add logging
    if config['logging']==True:
        # logger state
        logger_state = builder.AddSystem(VectorLogSink(plant.num_positions() + plant.num_velocities()))
        builder.Connect(
                plant.get_state_output_port(), 
                logger_state.get_input_port())
        # logger torque input
        logger_torque = builder.AddSystem(VectorLogSink(plant.num_actuators()))
        builder.Connect(
                plant.get_net_actuation_output_port(),  
                logger_torque.get_input_port())
        # logger joystick
        logger_joy = builder.AddSystem(VectorLogSink(5))
        builder.Connect(
                joystick.get_output_port(),
                logger_joy.get_input_port())
        # logger disturbances
        logger_distrubances = builder.AddSystem(VectorLogSink(plant.num_velocities()))
        builder.Connect(
                dist_gen.get_output_port(),
                logger_distrubances.get_input_port())
        # logger contact forces
        csv_file_name = "./data/data_contacts.csv"
        logger_contact = builder.AddSystem(ContactLogger(plant, sim_time_step, csv_file_name))    
        builder.Connect(
            plant.get_contact_results_output_port(),
            logger_contact.get_input_port(0))
    
    # Connect the plant to meshcat for visualization
    vis_config = VisualizationConfig()
    vis_config.publish_contacts = False
    ApplyVisualizationConfig(config=vis_config, builder=builder, meshcat=meshcat)

    # Set the meshcat position and target
    meshcat.SetCameraPose(
        config['camera']['position'],
        config['camera']['target']
    )

    # Build the system diagram
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    # Set the initial state
    q0 = standing_position()
    v0 = np.array(config['v0'])
    plant.SetPositions(plant_context, q0)
    plant.SetVelocities(plant_context, v0)
        
    # Change link masses
    if config['model_error']['enabled']:
        for body in plant.GetBodyIndices(plant.GetModelInstanceByName("achilles")):
            mass = plant.get_body(body).default_mass()
            r = np.random.normal(
                config['model_error']['mu'], config['model_error']['sigma'])
            new_mass = max(mass * r, 1e-4)
            plant.get_body(body).SetMass(plant_context, new_mass)


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

    if config['logging']==True:
        # unpack recorded data from the logger
        state_log = logger_state.FindLog(diagram_context)
        torque_log = logger_torque.FindLog(diagram_context)
        joy_log = logger_joy.FindLog(diagram_context)
        disturbance_log = logger_distrubances.FindLog(diagram_context)
        
        times = state_log.sample_times()
        states = state_log.data().T
        torques = torque_log.data().T
        joystick_commands = joy_log.data().T
        disturbances = disturbance_log.data().T

        # save the time data to a CSV file
        with open('./data/data_times.csv', mode='w') as file:
            writer = csv.writer(file)
            for i in range(len(times)):
                writer.writerow([times[i]])

        # save the state data to a CSV file
        with open('./data/data_states.csv', mode='w') as file:
            writer = csv.writer(file)
            for i in range(len(times)):
                writer.writerow(states[i])

        # save the torque data to a CSV file
        with open('./data/data_torques.csv', mode='w') as file:
            writer = csv.writer(file)
            for i in range(len(times)):
                writer.writerow(list(torques[i]))

        # save the joystick data to a CSV file
        with open('./data/data_joystick.csv', mode='w') as file:
            writer = csv.writer(file)
            for i in range(len(times)):
                writer.writerow(list(joystick_commands[i]))

        # save the disturbance data to a CSV file
        with open('./data/data_disturbances.csv', mode='w') as file:
            writer = csv.writer(file)
            for i in range(len(times)):
                writer.writerow(list(disturbances[i]))

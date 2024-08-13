#!/usr/bin/env python3

from pydrake.all import *
import numpy as np
import scipy as sp
import time
import math
import matplotlib.pyplot as plt

class HLIPTrajectoryGeneratorSE2(LeafSystem):

    # constructor
    def __init__(self, model_file):

        # init leaf system
        LeafSystem.__init__(self)

        # create internal model of the robot
        self.plant = MultibodyPlant(0)
        Parser(self.plant).AddModels(model_file)
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        # relevant frames
        self.static_com_frame = self.plant.GetFrameByName("static_com") # nominal is 0.734 z in world frame
        self.left_foot_frame = self.plant.GetFrameByName("left_foot")
        self.right_foot_frame = self.plant.GetFrameByName("right_foot")
        self.stance_foot_frame = None
        self.swing_foot_frame = None

        self.p_control_stance_W = np.array([0, 0, 0]).reshape(3,1)
        self.p_swing_init_W = np.array([0, 0, 0]).reshape(3,1)

        # walking parameters
        self.z_nom = 0.64  # nominal height of the CoM
        self.T_SSP = 0.3   # single support phase
        self.T_DSP = 0.0   # double support phase
        self.T = self.T_SSP + self.T_DSP

        # total horizon parameters, number S2S steps
        self.dt = 0.0
        self.N = 0
        self.T_horizon = 0.0
        self.T_leftover = False
        self.num_full_steps = 0

        # HLIP parameters
        g = 9.81
        self.lam = np.sqrt(g/self.z_nom)       # natural frequency
        self.A = np.array([[0,           1],   # LIP drift matrix
                           [self.lam**2, 0]])

        # create lambda function for hyperbolic trig
        coth = lambda x: (np.exp(2 * x) + 1) / (np.exp(2 * x) - 1)

        # define the deadbeat gains and orbital slopes
        self.Kp_db = 1
        self.Kd_db = self.T_DSP + (1/self.lam) * coth(self.lam * self.T_SSP)  # deadbeat gains      
        self.sigma_P1 = self.lam * coth(0.5 * self.lam * self.T_SSP)          # orbital slope (P1)
        self.v_des = 0.0

        # ROM HLIP state
        self.p_H_minus = None
        self.v_H_minus = None

        # swing foot parameters
        self.z_apex = 0.1     # height of the apex of the swing foot 
        self.z_offset = 0.05   # offset of the swing foot from the ground
        self.z0_offset = 0.0
        self.zf_offset = 0.0

        # bezier curve
        self.bezier_curve = None
        self.bez_order = 7  # 5 or 7

        # instantiate the inverse kinematics solver
        self.ik = InverseKinematics(self.plant)

        # inverse kinematics solver settings
        epsilon_feet = 0.00         # foot position tolerance      [m]
        epsilon_base = 0.00         # base position tolerance      [m]
        foot_epsilon_orient = 0.0   # foot orientation tolerance   [deg]
        base_epsilon_orient = 0.0   # torso orientation tolerance  [deg]
        self.tol_base = np.array([[epsilon_base], [np.inf], [epsilon_base]])
        self.tol_feet = np.array([[epsilon_feet], [np.inf], [epsilon_feet]]) 

        # Add com position constraint (updated at every S2S transition)
        self.p_com_cons = self.ik.AddPositionConstraint(self.static_com_frame, [0, 0, 0], 
                                                        self.plant.world_frame(), 
                                                        [0, 0, 0], [0, 0, 0])
        
        # Add com orientation constraint (fixed constraint)
        self.r_com_cons = self.ik.AddOrientationConstraint(self.static_com_frame, RotationMatrix(),
                                                           self.plant.world_frame(), RotationMatrix(),
                                                           base_epsilon_orient * (np.pi/180))

        # Add foot position constraints (continuously update the lower and upper bounds)
        self.p_left_cons =  self.ik.AddPositionConstraint(self.left_foot_frame, [0, 0, 0],
                                                          self.plant.world_frame(), 
                                                          [0, 0, 0], [0, 0, 0])
        self.p_right_cons = self.ik.AddPositionConstraint(self.right_foot_frame, [0, 0, 0],
                                                          self.plant.world_frame(), 
                                                          [0, 0, 0], [0, 0, 0]) 
        
        # Add foot orientation constraints (fixed constraint)
        self.r_left_cons =  self.ik.AddOrientationConstraint(self.left_foot_frame, RotationMatrix(),
                                                             self.plant.world_frame(), RotationMatrix(),
                                                             foot_epsilon_orient * (np.pi/180))
        self.r_right_cons = self.ik.AddOrientationConstraint(self.right_foot_frame, RotationMatrix(),
                                                             self.plant.world_frame(), RotationMatrix(),
                                                             foot_epsilon_orient * (np.pi/180))

    # -------------------------------------------------------------------------------------------------- #

    # update the COM state of the HLIP (need this incase you want to change v_des)
    def update_hlip_state_H(self):
        
        # Eq (20) Xiaobing Xiong, Ames
        T = self.T_SSP + self.T_DSP
        self.p_H_minus = (self.v_des * T) / (2 + self.T_DSP * self.sigma_P1)
        self.v_H_minus = self.sigma_P1 * (self.v_des * T) / (2 + self.T_DSP * self.sigma_P1)

    # switch the foot role positions
    def switch_foot_roles(self):
        
        # switch the stance and swing foot frames
        if self.stance_foot_frame == self.left_foot_frame:
            self.stance_foot_frame = self.right_foot_frame
            self.swing_foot_frame = self.left_foot_frame
        else:
            self.stance_foot_frame = self.left_foot_frame
            self.swing_foot_frame = self.right_foot_frame

    # -------------------------------------------------------------------------------------------------- #

    # set the trajectory generation problem parameters
    def set_problem_params(self, q0, v0, initial_stance_foot, v_des, dt, N):

        # make sure that N is non-zero
        assert N > 0, "N must be an integer greater than 0."

        # check if T_SSP is a multiple of dt (desirable)
        result = self.T_SSP / dt
        is_divisible = abs(round(result) - result) < 1e-6
        msg = "T_SSP must be a multiple of dt. You have T_SSP = {} and dt = {}".format(self.T_SSP, dt)
        assert is_divisible, msg

        # set the total horizon length
        self.dt = dt
        self.N = N
        self.T_horizon = N * dt

        # set the number of S2S steps
        self.num_full_steps = math.floor(self.T_horizon / self.T_SSP)

        # check if there is leftover time
        self.T_leftover = (self.T_horizon - self.num_full_steps * self.T_SSP) > 1e-6

        # set the robot state
        self.plant.SetPositions(self.plant_context, q0)
        self.plant.SetVelocities(self.plant_context, v0)

        # set the desired velocity
        self.v_des = v_des

        # set the initial stance foot
        if initial_stance_foot == "left_foot":
            self.stance_foot_frame = self.left_foot_frame
            self.swing_foot_frame = self.right_foot_frame

        elif initial_stance_foot == "right_foot":
            self.stance_foot_frame = self.right_foot_frame
            self.swing_foot_frame = self.left_foot_frame

        # set the intial stance foot control frame position in world frame
        p_stance_W = self.plant.CalcPointsPositions(self.plant_context,
                                                    self.stance_foot_frame,
                                                    [0,0,0],
                                                    self.plant.world_frame())
        self.p_control_stance_W = np.array([p_stance_W[0], p_stance_W[1], [0]])

        # set the initial swing foot position in world frame
        self.p_swing_init_W = self.plant.CalcPointsPositions(self.plant_context,
                                                             self.swing_foot_frame,
                                                             [0,0,0],
                                                             self.plant.world_frame())

    # -------------------------------------------------------------------------------------------------- #

    # compute the times intervals of the executions, I
    def compute_execution_intervals(self):

        # compute the execution time intervals, I
        I = []
        if (self.num_full_steps == 0) and (self.T_leftover == True):
            time_set = np.linspace(0, self.T_horizon - self.dt, int(self.T_horizon / self.dt))
            I.append(time_set)
        
        elif (self.num_full_steps > 0) and (self.T_leftover == False):
            for _ in range(self.num_full_steps):
                time_set = np.linspace(0, self.T_SSP - self.dt, int(self.T_SSP / self.dt))
                I.append(time_set)
        
        elif (self.num_full_steps > 0) and (self.T_leftover == True):
            for _ in range(self.num_full_steps):
                time_set = np.linspace(0, self.T_SSP - self.dt, int(self.T_SSP / self.dt))
                I.append(time_set)

            n_leftover = int(self.N - self.num_full_steps * int(self.T_SSP / self.dt))
            times_set = np.linspace(0, self.dt * (n_leftover-1), n_leftover)    
            I.append(times_set)   

        return I
    
    # compute the continous solution trajectories, C
    def compute_execution_solutions(self, L, I):

        # compute the HLIP preimpact state
        self.p_H_minus = (self.v_des * self.T) / (2 + self.T_DSP * self.sigma_P1)
        self.v_H_minus = self.sigma_P1 * (self.v_des * self.T) / (2 + self.T_DSP * self.sigma_P1)

        # compute the current COM state of the robot
        p_com_W = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context).reshape(3,1)

        # compute the current COM velocity of the robot
        J = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(self.plant_context, 
                                                                     JacobianWrtVariable.kV,
                                                                     self.plant.world_frame(),
                                                                     self.plant.world_frame())
        v = self.plant.GetVelocities(self.plant_context)
        v_com_W = (J @ v).reshape(3,1)

        # compute the intial state of the Robot LIP model
        p_R = p_com_W[0][0] - self.p_control_stance_W[0][0]
        v_R = v_com_W[0][0]

        # update the HLIP preimpact state
        self.update_hlip_state_H()

        # compute the flow of the Robot following LIP dynamics
        p_stance_pos_list = [self.p_control_stance_W]
        p_stance_name_list = [self.stance_foot_frame.name()]
        C = []

        for k in range(len(L)):
             
            # inital condition and time set
            x0 = np.array([p_R, v_R]).reshape(2,1)
            time_set = I[k]

            # compute the flow of the LIP model
            xt_list = []
            for t in time_set:
                xt = sp.linalg.expm(self.A * t) @ x0
                xt_list.append(xt)

            # append to the list of continuous solutions
            C.append(xt_list)

            # compute the preimpact state of the Robot LIP model
            xt_minus = sp.linalg.expm(self.A * (self.T_SSP)) @ x0
            p_R_minus = xt_minus[0][0]
            v_R_minus = xt_minus[1][0]

            # compute the swing foot target position relative to control frame
            u = self.v_des * self.T_SSP + self.Kp_db * (p_R_minus - self.p_H_minus) + self.Kd_db * (v_R_minus - self.v_H_minus)
            
            # add to the list of stance foot postions
            p_swing_target = self.p_control_stance_W + np.array([u, 0, 0]).reshape(3,1)
            p_stance_pos_list.append(p_swing_target)

            # switch the foot roles
            self.p_control_stance_W = p_swing_target
            self.switch_foot_roles()
            p_stance_name_list.append(self.stance_foot_frame.name())

            # set the new intial condition
            p_R = p_R_minus - u
            v_R = v_R_minus

        return C, p_stance_pos_list, p_stance_name_list

    # precompute the execution of the LIP model in world frame, X = (Lambda, I, C)
    def compute_LIP_execution(self):

        # compute the execution time intervals, I
        I = self.compute_execution_intervals()

        # compute the execution index, Lambda
        L = np.arange(0, len(I))

        # compute the continuous solution trajectories, C
        C, p_stance_pos_list, p_stance_name_list = self.compute_execution_solutions(L, I)

        # plot the C trajcetories
        # print(C)
        # for i in range(len(C)):
        #     traj = C[i]
        #     print(traj)
        #     print(len(traj))
        #     for j in range(len(traj)):
        #         plt.plot(traj[j][0], traj[j][1], 'ro')
        # plt.show()

        return (L, I, C), p_stance_pos_list, p_stance_name_list
            
    # -------------------------------------------------------------------------------------------------- #

    # main function that updates the whole problem
    def get_trajectory(self, q0, v0, initial_stance_foot, v_des, dt, N):
        
        # setup the problem parameters
        self.set_problem_params(q0, v0, initial_stance_foot, v_des, dt, N)

        # compute the LIP execution in world frame, X = (Lambda, I, C)
        L, I, C = self.compute_LIP_execution()

        exit()

        # # for every swing foot configuration solve the IK problem
        # q_ik_sol_list = []
        # q_ik_sol = None
        # for i in range(n):
            
        #     # set the initial guess of the IK problem
        #     if i == 0:
        #         q_guess = q0
        #     else:
        #         q_guess = q_ik_sol
        #     # q_guess = q0

        #     # solve the IK problem
        #     res = self.solve_ik(self.p_control_stance_W, swing_traj[i], q_guess)
            
        #     if res.is_success():
        #         q_ik_sol = res.GetSolution()
        #         q_ik_sol_list.append(q_ik_sol)
        #     else:
        #         print("IK failed at time: {}, {}".format(i, times[i]))
        #         break

        # return the trajectory
        return q_ik_sol_list

######################################################################################################################

if __name__ == "__main__":

    # model path
    model_file = "../../models/achilles_SE2_drake.urdf"

    # create the trajectory generator
    g = HLIPTrajectoryGeneratorSE2(model_file)
    
    # set desired problem parameters
    v_des = 0.0
    stance_foot = "right_foot"
    q0 = np.array([0, 0.96,   # position (x,z)
                   0.1,        # theta
                   -0.63, 0.34, 0.17,  # left leg: hip_pitch, knee, ankle 
                   -0.22, 0.69, -0.54]) # right leg: hip_pitch, knee, ankle
    v0 = np.zeros(len(q0))
    v0[0] = -0.0              # forward x-velocity

    t0 = time.time()
    q_ik_list = g.get_trajectory(q0 = q0, 
                                 v0 = v0,
                                 initial_stance_foot = stance_foot,
                                 v_des = v_des,
                                 dt = 0.01,
                                 N = 59)
    print("Time to solve the IK problem: ", time.time() - t0)
    print("Average time per IK problem: ", (time.time() - t0) / len(q_ik_list))

    # start meshcat
    meshcat = StartMeshcat()

    # Set up a system diagram that includes a plant, scene graph, and meshcat
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    models = Parser(plant).AddModels(model_file)
    plant.Finalize()

    AddDefaultVisualization(builder, meshcat)

    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    # Start meshcat recording
    meshcat.StartRecording()

    time_elapsed = 0.0
    tot_time_des = 5.0
    configs_per_sec = len(q_ik_list) / tot_time_des
    dt = 1.0 / configs_per_sec
    for i in range(len(q_ik_list)):

        # Wait for the next state estimate        
        time.sleep(dt)

        # Set the Drake model to have this state
        q0 = q_ik_list[i]
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





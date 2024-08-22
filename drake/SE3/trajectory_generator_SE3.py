#!/usr/bin/env python3

from pydrake.all import *
import numpy as np
import scipy as sp
import time

class HLIPTrajectoryGeneratorSE3():

    # constructor
    def __init__(self, model_file):

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

        self.control_stance_yaw = None
        self.p_swing_init_W = None
        self.p_control_stance_W = None
        self.R_control_satnce_W = None
        self.R_control_stance_W_mat = None

        # total horizon parameters, number S2S steps
        self.dt = None
        self.N = None

        # for keeping track of the pahse time
        self.T_DSP = None
        self.T_SSP = None
        self.T = None
        self.t_phase = None

        # create lambda function for hyperbolic trig
        self.coth = lambda x: (np.exp(2 * x) + 1) / (np.exp(2 * x) - 1)
        self.tanh = lambda x: (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        self.sech = lambda x: 1 / np.cosh(x)

        # swing foot parameters
        self.z_nom = 0.64
        self.z_apex = 0.08     # height of the apex of the swing foot 
        self.z_offset = 0.0    # offset of the swing foot from the ground
        self.z0_offset = 0.0
        self.zf_offset = 0.0

        # clip the swing foot target position
        self.ux_max = 0.4
        self.uy_max = 0.4

        # maximum velocity
        self.vx_des = None
        self.vy_des = None

        # period 2 feedforward foot placements
        self.u_L_bias = 0.28   # left is swing foot, add this to the feedforward foot placement
        self.u_R_bias = -0.28  # right is swing foot, add this to the feedforward foot placement
        self.u_L = None
        self.u_R = None
        
        # preimpact state
        self.lam = None

        # bezier curve
        self.bez_order = 7  # 5 or 7

        # instantiate the inverse kinematics solver
        self.ik = InverseKinematics(self.plant, with_joint_limits=True)

        # inverse kinematics solver settings
        epsilon_feet = 0.00         # foot position tolerance      [m]
        epsilon_base = 0.00         # base position tolerance      [m]
        self.foot_epsilon_orient = 0.0   # foot orientation tolerance   [deg]
        self.base_epsilon_orient = 0.0   # torso orientation tolerance  [deg]
        self.tol_base = np.array([[epsilon_base], [epsilon_base], [epsilon_base]])
        self.tol_feet = np.array([[epsilon_feet], [epsilon_feet], [epsilon_feet]]) 

        # Add com position constraint (continusously update the lower and upper bounds)
        self.p_com_cons = self.ik.AddPositionConstraint(self.static_com_frame, [0, 0, 0], 
                                                        self.plant.world_frame(), 
                                                        [0, 0, 0], [0, 0, 0])

        # Add foot position constraints (continuously update the lower and upper bounds)
        self.p_left_cons =  self.ik.AddPositionConstraint(self.left_foot_frame, [0, 0, 0],
                                                          self.plant.world_frame(), 
                                                          [0, 0, 0], [0, 0, 0])
        self.p_right_cons = self.ik.AddPositionConstraint(self.right_foot_frame, [0, 0, 0],
                                                          self.plant.world_frame(), 
                                                          [0, 0, 0], [0, 0, 0]) 
        
        # Add foot and com orientation constraints, aligns toe directions (updated once at every trajectory gen query)
        self.r_com_cons = self.ik.AddOrientationConstraint(self.static_com_frame, RotationMatrix(), 
                                                           self.plant.world_frame(), RotationMatrix(), 
                                                           0.0)
        self.r_left_cons = self.ik.AddAngleBetweenVectorsConstraint(self.left_foot_frame, [1, 0, 0],
                                                                    self.plant.world_frame(), [1, 0, 0],
                                                                    0, 0)
        self.r_right_cons = self.ik.AddAngleBetweenVectorsConstraint(self.right_foot_frame, [1, 0, 0],
                                                                    self.plant.world_frame(), [1, 0, 0],
                                                                    0, 0)

    # -------------------------------------------------------------------------------------------------- #

    # rotate local stance foot control frame step increment to increment in world frame
    def RotateVectorToWorld(self, vx_local, vy_local):

        # rotate the step increment to world frame
        R_z = np.array([[np.cos(self.control_stance_yaw), -np.sin(self.control_stance_yaw)],
                        [np.sin(self.control_stance_yaw), np.cos(self.control_stance_yaw)]])
        v_world = R_z @ np.array([[vx_local], [vy_local]])
        vx_world = v_world[0][0]
        vy_world = v_world[1][0]

        return vx_world, vy_world

    # -------------------------------------------------------------------------------------------------- #

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

    # compute the bezier curve for the swing foot trajectory
    def compute_bezier_curve(self, swing_foot_pos_init_W, swing_foot_target_W):

        # initial and final positions
        u0_x = swing_foot_pos_init_W[0][0]
        u0_y = swing_foot_pos_init_W[1][0]
        uf_x = swing_foot_target_W[0][0]
        uf_y = swing_foot_target_W[1][0]

        # compute primary bezier curve control points
        if self.bez_order == 7:
            ctrl_pts_x = np.array([u0_x, u0_x, u0_x, (u0_x+uf_x)/2, uf_x, uf_x, uf_x])
            ctrl_pts_y = np.array([u0_y, u0_y, u0_y, (u0_y+uf_y)/2, uf_y, uf_y, uf_y])
            ctrl_pts_z = np.array([self.z0_offset, self.z0_offset, self.z0_offset, (16/5)*(self.z_apex), self.zf_offset, self.zf_offset, self.zf_offset]) + self.z_offset

        elif self.bez_order == 5:
            ctrl_pts_x = np.array([u0_x, u0_x, (u0_x+uf_x)/2, uf_x, uf_x])
            ctrl_pts_y = np.array([u0_y, u0_y, (u0_y+uf_y)/2, uf_y, uf_y])
            ctrl_pts_z = np.array([self.z0_offset, self.z0_offset, (8/3)*self.z_apex, self.zf_offset, self.zf_offset]) + self.z_offset

        # set the primary control points
        ctrl_pts = np.vstack((ctrl_pts_x, 
                              ctrl_pts_y,
                              ctrl_pts_z))

        # create the bezier curve
        b = BezierCurve(0, self.T_SSP, ctrl_pts)

        return b

    # -------------------------------------------------------------------------------------------------- #

    # compute the times intervals of the executions, I
    def compute_execution_intervals(self):

        # compute the execution time intervals, I. NOTE: using round to avoid floating point errors
        I = []
        time_set = []
        t = self.t_phase

        for _ in range(self.N):
        
            time_set.append(t)
            if round(t + self.dt, 5) < self.T_SSP:
                t = round(t + self.dt, 5)
            else:
                I.append(time_set)
                time_set = []
                t = round(t + self.dt, 5) - self.T_SSP
        
        if len(time_set) > 0:
            I.append(time_set)

        return I

    # compute the continous solution trajectories, C
    def compute_execution_solutions(self, L, I):

        # update the (P1) preimpact state for x-direction
        px_H_minus = (self.vx_des * self.T) / (2 + self.T_DSP * self.sigma_P1)
        vx_H_minus = self.sigma_P1 * (self.vx_des * self.T) / (2 + self.T_DSP * self.sigma_P1)

        # compute the current COM state of the robot
        p_com_W = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context).reshape(3,1)
        p_com_control_stance = self.R_control_stance_W_mat.T @ (p_com_W - self.p_control_stance_W)

        # compute the current COM velocity of the robot
        J = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(self.plant_context, 
                                                                     JacobianWrtVariable.kV,
                                                                     self.plant.world_frame(),
                                                                     self.plant.world_frame())
        v = self.plant.GetVelocities(self.plant_context)
        v_com_W = (J @ v).reshape(3,1)
        v_com_control_stance = self.R_control_stance_W_mat.T @ v_com_W

        # compute the intial state of the Robot LIP model in stance frame
        px_R = p_com_control_stance[0][0]
        py_R = p_com_control_stance[1][0]
        vx_R = v_com_control_stance[0][0]
        vy_R = v_com_control_stance[1][0]

        # compute the flow of the Robot following LIP dynamics
        stance_name_list = []  # name of the stance foot frame
        p_foot_pos_list = []   # each elemnt is a 3-tuple (p_stance, p_swing_init, p_swing_target)
        C = []                 # list of continuous solutions
        for k in range(len(L)):

            # inital condition and time set
            x0 = np.array([px_R, vx_R]).reshape(2,1)
            y0 = np.array([py_R, vy_R]).reshape(2,1)
            time_set = I[k]

            # compute the flow of the LIP model
            xt_list = []
            for t in time_set:
                xt = sp.linalg.expm(self.A * (t- time_set[0])) @ x0
                yt = sp.linalg.expm(self.A * (t- time_set[0])) @ y0
                sol = np.vstack((xt, yt))
                xt_list.append(sol)

            # append to the list of continuous solutions
            C.append(xt_list)

            # compute the preimpact state of the Robot LIP model
            xt_minus = sp.linalg.expm(self.A * (self.T_SSP - time_set[0])) @ x0
            yt_minus = sp.linalg.expm(self.A * (self.T_SSP - time_set[0])) @ y0
            px_R_minus = xt_minus[0][0]
            vx_R_minus = xt_minus[1][0]
            py_R_minus = yt_minus[0][0]
            vy_R_minus = yt_minus[1][0]

            # compute the HLIP preimpact state (P2 orbit dependent in y-direction)
            if self.swing_foot_frame.name() == "left_foot":
                u_star = self.u_L
            elif self.swing_foot_frame.name() == "right_foot":
                u_star = self.u_R
            py_H_minus = (u_star - self.T_DSP * self.d2) / (2 + self.T_DSP * self.sigma_P2)
            vy_H_minus = self.sigma_P2 * py_H_minus + self.d2

            ux = self.vx_des * self.T_SSP + self.Kp_db * (px_R_minus - px_H_minus) + self.Kd_db * (vx_R_minus - vx_H_minus)
            uy = self.vy_des * self.T_SSP + self.Kp_db * (py_R_minus - py_H_minus) + self.Kd_db * (vy_R_minus - vy_H_minus)

            if self.swing_foot_frame.name() == "left_foot":
                uy += self.u_L
            elif self.swing_foot_frame.name() == "right_foot":
                uy += self.u_R
            
            # clip the swing foot target position
            ux = max(-self.ux_max, min(ux, self.ux_max))
            uy = max(-self.uy_max, min(uy, self.uy_max))
            ux_W, uy_W = self.RotateVectorToWorld(ux, uy)

            # populate foot position information
            if k == 0:
                p_stance = self.p_control_stance_W
                p_swing_init = self.p_swing_init_ground
                p_swing_target = p_stance + np.array([ux_W, uy_W, 0]).reshape(3,1)
            else:
                p_stance_temp = p_stance
                p_stance = p_swing_target
                p_swing_init = p_stance_temp
                p_swing_target = p_stance + np.array([ux_W, uy_W, 0]).reshape(3,1)
            p_foot_pos_info = (p_stance, p_swing_init, p_swing_target)
            p_foot_pos_list.append(p_foot_pos_info)
            stance_name_list.append(self.stance_foot_frame.name())

            # update the feet position info NOTE: I need to reason about the y-direction at some point
            self.switch_foot_roles()
            self.p_control_stance_W = np.array([p_swing_target[0], p_swing_target[1], [0]])
            
            if k == 0:
                self.p_swing_init_W = p_stance
            else:
                self.p_swing_init_W = p_stance_temp

            # prepare for the intial condition of the next time step
            if k < len(L) - 1:

                # compute the postimpact state of the Robot LIP model
                px_R = px_R_minus - ux
                vx_R = vx_R_minus
                
                py_R = py_R_minus - uy
                vy_R = vy_R_minus

                # forward prop up to the first time in the next time set
                x0 = np.array([px_R, vx_R]).reshape(2,1)
                xt = sp.linalg.expm(self.A * (I[k+1][0])) @ x0
                px_R = xt[0][0]
                vx_R = xt[1][0]
                
                y0 = np.array([py_R, vy_R]).reshape(2,1)
                yt = sp.linalg.expm(self.A * (I[k+1][0])) @ y0
                py_R = yt[0][0]
                vy_R = yt[1][0]

        return C, p_foot_pos_list, stance_name_list

    # precompute the execution of the LIP model in world frame, X = (Lambda, I, C)
    def compute_LIP_execution(self):

        # compute the execution time intervals, I
        I = self.compute_execution_intervals()

        # compute the execution index, Lambda
        L = np.arange(0, len(I))

        # compute the continuous solution trajectories, C
        C, p_foot_pos_list, stance_name_list = self.compute_execution_solutions(L, I)

        # execution is three tuple, X = (L, I, C)
        X = (L, I, C)

        return X, p_foot_pos_list, stance_name_list

    # -------------------------------------------------------------------------------------------------- #

    # solve the inverse kinematics problem
    def solve_ik(self, p_com_pos, p_stance, p_swing, stance_name, initial_guess):

        # update the COM target position
        self.p_com_cons.evaluator().set_bounds(p_com_pos - self.tol_base, p_com_pos + self.tol_base)

        if stance_name == "left_foot":
            self.p_left_cons.evaluator().set_bounds(p_stance - self.tol_feet, p_stance + self.tol_feet)
            self.p_right_cons.evaluator().set_bounds(p_swing - self.tol_feet, p_swing + self.tol_feet)

        elif stance_name == "right_foot":
            self.p_right_cons.evaluator().set_bounds(p_stance - self.tol_feet, p_stance + self.tol_feet)
            self.p_left_cons.evaluator().set_bounds(p_swing - self.tol_feet, p_swing + self.tol_feet)

        # solve the IK problem
        self.ik.prog().SetInitialGuess(self.ik.q(), initial_guess)
        res = SnoptSolver().Solve(self.ik.prog())

        return res

    # -------------------------------------------------------------------------------------------------- #        

    # get hamiltonian product
    def quat_mult(self, q1, q2):

        # exatrct the quaternion components
        q1_w, q1_x, q1_y, q1_z = q1[0], q1[1], q1[2], q1[3]
        q2_w, q2_x, q2_y, q2_z = q2[0], q2[1], q2[2], q2[3]

        # mulitply the quaternions
        q_w = q1_w*q2_w - q1_x*q2_x - q1_y*q2_y - q1_z*q2_z
        q_x = q1_w*q2_x + q1_x*q2_w + q1_y*q2_z - q1_z*q2_y
        q_y = q1_w*q2_y - q1_x*q2_z + q1_y*q2_w + q1_z*q2_x
        q_z = q1_w*q2_z + q1_x*q2_y - q1_y*q2_x + q1_z*q2_w

        return np.array([q_w, q_x, q_y, q_z])
    
    # get the quaternion conjugate
    def quat_conj(self, q):

        # extract the quaternion components
        q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]

        return np.array([q_w, -q_x, -q_y, -q_z])
    
    # get the Lie algebra from a quaternion
    def quat_to_lie_algebra(self, q):

        # extract the quaternion components
        q_w, q_x, q_y, q_z = q[0], q[1], q[2], q[3]

        # get scalar and vector parts
        w = q_w
        v = np.array([q_x, q_y, q_z])

        # get the Lie algebra elements for sufficiently big quaternion difference
        if np.linalg.norm(v) < 1e-6:
            s = np.array([0, 0, 0])
        else:
            theta = 2 * np.arctan2(np.linalg.norm(v), w)
            s = theta * v / np.linalg.norm(v)

        return s
    
    # get omega given q1 and q2 (q1 is older and q2 is newer in time)
    def compute_omega(self, q1, q2):

        # the error quaternion
        qe = self.quat_mult(self.quat_conj(q1), q2)
        ds = self.quat_to_lie_algebra(qe)
        omega = ds / self.dt
        
        return omega

    # compute the velocity reference
    def compute_velocity_reference(self, q_ref, v0):

        # do finite difference, v_k = (q_k - q_k-1) / dt
        v_ref = []
        for k in range(len(q_ref)):
            
            # finite difference
            if k < len(q_ref) - 1:
                # handle the quaternion velocities
                quat1 = q_ref[k][:4]
                quat2 = q_ref[k+1][:4]
                omega_k = self.compute_omega(quat1, quat2)

                # handel all other euclidean velocities
                vel_k = (q_ref[k+1][4:] - q_ref[k][4:]) / self.dt

                # combine them and add to list
                v_k = np.concatenate((omega_k, vel_k))
                v_ref.append(v_k)
            
            # last velocity is the same as the previous one
            else:
                v_ref.append(v_k)

        return v_ref

    # -------------------------------------------------------------------------------------------------- #

    # set the trajectory generation problem parameters
    def set_parameters(self, z_nom, z_apex, bezier_order, T_SSP, dt, N):

        # make sure that N is non-zero
        assert N >= 1, "N must be an integer >= 1."

        # set time paramters
        self.T_SSP = T_SSP
        self.T_DSP = 0.0   # double support phase
        self.T = self.T_SSP + self.T_DSP

        # set the total horizon length
        self.dt = dt
        self.N = N

        # set bezier curve order
        self.bez_order = bezier_order

        # set variables that depend on z com nominal
        self.z_apex = z_apex
        self.z_nom = z_nom
        g = 9.81
        self.lam = np.sqrt(g/self.z_nom)       # natural frequency
        self.A = np.array([[0,           1],   # LIP drift matrix
                           [self.lam**2, 0]])

        # define the deadbeat gains and orbital slopes
        self.Kp_db = 1
        self.Kd_db = self.T_DSP + (1/self.lam) * self.coth(self.lam * self.T_SSP)  # deadbeat gains
        self.sigma_P1 = self.lam * self.coth(0.5 * self.lam * self.T_SSP)          # orbital slope (P1)
        self.sigma_P2 = self.lam * self.tanh(0.5 * self.lam * self.T_SSP)          # orbital slope (P2)

        return "Parameters set successfully."

    # -------------------------------------------------------------------------------------------------- #

    # main function that updates the whole problem
    def generate_trajectory(self, q0, v0, v_des, t_phase, initial_swing_foot_pos, stance_foot_pos, stance_foot_yaw, initial_stance_foot_name):

        # set the robot state
        self.plant.SetPositions(self.plant_context, q0)
        self.plant.SetVelocities(self.plant_context, v0)

        # set the desired velocity
        self.vx_des = v_des[0][0]
        self.vy_des = v_des[1][0]

        # set the P2 orbit bias
        self.u_L = self.u_L_bias + self.vy_des * (self.T_SSP + self.T_DSP)
        self.u_R = self.u_R_bias + self.vy_des * (self.T_SSP + self.T_DSP)

        # P2 orbit shifting
        self.d2 = self.lam**2 * (self.sech(0.5 * self.lam * self.T_SSP))**2 * (self.T * self.vy_des) / (self.lam**2 * self.T_DSP + 2 * self.sigma_P2)

        # set the intial phase
        self.t_phase = t_phase

        # set the initial stance foot
        if initial_stance_foot_name == "left_foot":
            self.stance_foot_frame = self.left_foot_frame
            self.swing_foot_frame = self.right_foot_frame

        elif initial_stance_foot_name == "right_foot":
            self.stance_foot_frame = self.right_foot_frame
            self.swing_foot_frame = self.left_foot_frame

        # set the initial stance foot control frame position in world frame
        self.control_stance_yaw = stance_foot_yaw
        self.p_control_stance_W = np.array([stance_foot_pos[0], stance_foot_pos[1], [0]])
        self.R_control_stance_W = RotationMatrix(RollPitchYaw(0, 0, stance_foot_yaw))
        self.R_control_stance_W_mat = self.R_control_stance_W.matrix()

        # set the stance foot yaw
        self.control_stance_yaw = stance_foot_yaw

        # set the initial swing foot position in world frame
        self.p_swing_init_ground = initial_swing_foot_pos           # swing foot pos when it takes off from the ground

        # update the foot and torso orientation constraints
        self.ik.prog().RemoveConstraint(self.r_com_cons)
        self.ik.prog().RemoveConstraint(self.r_left_cons)
        self.ik.prog().RemoveConstraint(self.r_right_cons)
        self.r_com_cons = self.ik.AddOrientationConstraint(self.static_com_frame, RotationMatrix(),
                                                           self.plant.world_frame(), self.R_control_stance_W,
                                                           self.base_epsilon_orient)
        self.r_left_cons = self.ik.AddAngleBetweenVectorsConstraint(self.left_foot_frame, [1, 0, 0],
                                                                    self.plant.world_frame(), self.R_control_stance_W_mat @ [1, 0, 0],
                                                                    0, self.foot_epsilon_orient * np.pi / 180)
        self.r_right_cons = self.ik.AddAngleBetweenVectorsConstraint(self.right_foot_frame, [1, 0, 0],
                                                                    self.plant.world_frame(), self.R_control_stance_W_mat @ [1, 0, 0],
                                                                    0, self.foot_epsilon_orient * np.pi / 180)

        # compute the LIP execution in world frame, X = (Lambda, I, C)
        # t0 = time.time()
        X, p_foot_pos_list, stance_name_list = self.compute_LIP_execution()
        L, I, C = X[0], X[1], X[2]
        # tf = time.time()
        # print("Time to compute LIP execution: ", tf - t0)

        # for every swing foot configuration solve the IK problem
        q_ref = []
        q_ik_sol = q0
        # t0 = time.time()
        for i in L:

            # unpack the foot position information tuple
            p_stance, p_swing_init, p_swing_target = p_foot_pos_list[i]
            stance_foot_name = stance_name_list[i]

            # unpack the time_set
            time_set = I[i]

            # unpack the continuous solution
            xt_R = C[i]

            # compute the bezier curve for the swing foot trajectory
            b_swing = self.compute_bezier_curve(p_swing_init, p_swing_target)

            # solve the IK problem
            for t, k in zip(time_set, range(len(time_set))):
                
                # compute the swing foot target
                b_t = b_swing.value(t)
                p_swing_target_W = np.array([b_t[0], b_t[1], b_t[2]]) 

                # compute the COM position target
                px_R = xt_R[k][0][0]
                py_R = xt_R[k][2][0]
                px_R_W, py_R_W = self.RotateVectorToWorld(px_R, py_R)
                p_com_pos =  p_stance + np.array([px_R_W, py_R_W, self.z_nom]).reshape(3,1)

                res = self.solve_ik(p_com_pos, p_stance, p_swing_target_W, stance_foot_name, q_ik_sol)
                if res.is_success():
                    q_ik_sol = res.GetSolution(self.ik.q())
                    q_ref.append(q_ik_sol)
                else:
                    q_ref.append(q_ik_sol)
                    print("IK problem failed at time: {}, index {}".format(t, i))

        # get the velocity reference
        v_ref = self.compute_velocity_reference(q_ref, v0)

        # tf = time.time()
        # print("Time to solve IK: ", tf - t0)
        # print("Average time per IK: ", (tf - t0) / len(q_ref))

        # return the trajectory
        return q_ref, v_ref

####################################################################################################

if __name__ == "__main__":

    # model file
    model_file = "../../models/achilles_SE3_drake.urdf"

    # create a plant model for testing
    plant = MultibodyPlant(0)
    Parser(plant).AddModels(model_file)
    plant.Finalize()
    plant_context = plant.CreateDefaultContext()

    # create the trajectory generator
    traj_gen = HLIPTrajectoryGeneratorSE3(model_file)

    # set the parameters
    traj_gen.set_parameters(z_nom=0.64, 
                            z_apex=0.05, 
                            bezier_order=7, 
                            T_SSP=0.3, 
                            dt=0.02, 
                            N=400)

    # initial condition 
    q0 = np.array([
        1.0000, 0.0000, 0.0000, 0.0000,            # base orientation, (w, x, y, z)
        0.0000, 0.0000, 0.9300,                    # base position, (x,y,z)
        0.0000,  0.0209, -0.5515, 1.0239,-0.4725,  # left leg, (hip yaw, hip roll, hip pitch, knee, ankle) 
        0.0000, -0.0209, -0.5515, 1.0239,-0.4725,  # left leg, (hip yaw, hip roll, hip pitch, knee, ankle) 
    ])
    v0 = np.zeros(plant.num_velocities())
    v0[3] = 0.0 # x velocity
    v0[4] = 0.0 # forward velocity
    plant.SetPositions(plant_context, q0)
    plant.SetVelocities(plant_context, v0)

    swing_foot_frame = plant.GetFrameByName("left_foot")
    stance_foot_frame = plant.GetFrameByName("right_foot")

    p_swing = plant.CalcPointsPositions(plant_context,
                                        swing_foot_frame, [0, 0, 0],
                                        plant.world_frame())
    p_stance = plant.CalcPointsPositions(plant_context, 
                                         stance_foot_frame, [0, 0, 0], 
                                         plant.world_frame())
    R_stance = plant.CalcRelativeRotationMatrix(plant_context,
                                                plant.world_frame(),
                                                stance_foot_frame)
    yaw = RollPitchYaw(R_stance).yaw_angle()

    # generate a trajectory
    v_des = np.array([[0.], [-0.2]])
    t_phase = 0.0

    t0 = time.time()
    q_HLIP, v_HLIP = traj_gen.generate_trajectory(q0=q0,
                                                  v0=v0,
                                                  v_des=v_des,
                                                  t_phase=t_phase,
                                                  initial_swing_foot_pos=p_swing,
                                                  stance_foot_pos=p_stance,
                                                  stance_foot_yaw=yaw,
                                                  initial_stance_foot_name="right_foot")
    tf = time.time()
    print("Time to solve 3D prop: ", tf - t0)
    
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
    configs_per_sec = len(q_HLIP) / tot_time_des
    dt = 1.0 / configs_per_sec
    for i in range(len(q_HLIP)):

        # Wait for the next state estimate        
        time.sleep(dt)

        # Set the Drake model to have this state
        q0 = q_HLIP[i]
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

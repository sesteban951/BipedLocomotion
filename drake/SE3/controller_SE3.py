#!/usr/bin/env python
from pydrake.all import *
import numpy as np
import scipy as sp

class HLIP(LeafSystem):

    # constructor
    def __init__(self, model_file, meshcat):

        # init leaf system
        LeafSystem.__init__(self)

        # meshcat
        self.meshcat = meshcat

        # visualization colors and shapes
        self.red_color = Rgba(1, 0, 0, 1)
        self.green_color = Rgba(0, 1, 0, 1)
        self.blue_color = Rgba(0, 0, 1, 1)
        self.sphere = Sphere(0.025)
        cyl = Cylinder(0.005, 0.2)

        self.meshcat.SetObject("com_pos", self.sphere, self.green_color)
        self.meshcat.SetObject("p_right", self.sphere, self.red_color)
        self.meshcat.SetObject("p_left", self.sphere, self.blue_color)

        # stace foot frame visualization 
        self.meshcat.SetObject("stance_x", cyl, self.red_color)
        self.meshcat.SetObject("stance_y", cyl, self.green_color)
        self.meshcat.SetObject("stance_z", cyl, self.blue_color)

        # create internal model of the robot
        self.plant = MultibodyPlant(0)
        Parser(self.plant).AddModels(model_file)
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        # Leaf System input port
        self.input_port = self.DeclareVectorInputPort(
                            "x_hat", 
                            BasicVector(self.plant.num_positions() + self.plant.num_velocities()))
        self.gamepad_port = self.DeclareVectorInputPort(
                            "joy_command",
                            BasicVector(5))  # LS_x, LS_y, RS_x, A button (Xbox), RT
        self.DeclareVectorOutputPort(
                            "x_des",
                            BasicVector(2 * self.plant.num_actuators()),
                            self.CalcOutput)

        # relevant frames
        self.static_com_frame = self.plant.GetFrameByName("static_com") # nominal is 0.734 z in world frame
        self.left_foot_frame = self.plant.GetFrameByName("left_foot")
        self.right_foot_frame = self.plant.GetFrameByName("right_foot")
        self.p_control_stance_W = None
        self.R_control_stance_W = None
        self.control_stance_yaw = 0

        # Foot position Variables
        self.p_swing_init_W = None

        # for updating number of steps amd switching foot stance/swing
        self.update_foot_role_flag = True
        self.num_steps = 0

        # walking parameters
        self.z_nom = 0.64  # nominal height of the CoM
        self.T_SSP = 0.30  # single support phase
        self.T_DSP = 0.0   # double support phase

        # HLIP parameters
        g = 9.81
        self.lam = np.sqrt(g/self.z_nom)
        self.A = np.array([[0,           1],   # LIP drift matrix
                           [self.lam**2, 0]])

        # Robot LIP state
        self.p_R = None     # center of mass position (in control stance foot frame)
        self.v_R = None     # center of mass velocity (in control stance foot frame)

        # LIP model preimpact states
        self.p_H_minus_x = 0
        self.v_H_minus_x = 0
        self.p_R_minus_x = 0
        self.v_R_minus_x = 0
        self.p_H_minus_y = 0
        self.v_H_minus_y = 0
        self.p_R_minus_y = 0
        self.v_R_minus_y = 0

        # swing foot parameters
        self.z_apex = 0.08    # NOTE: this is affected by bezier curve swing belnding
        self.z_offset = 0.0
        self.z0 = 0.0
        self.zf = 0.0

        # lambda functions for hyperbolic trig
        coth = lambda x: (np.exp(2 * x) + 1) / (np.exp(2 * x) - 1)
        tanh = lambda x: (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        self.sech = lambda x: 1 / np.cosh(x)

        # define deadbeat gains and orbital slopes
        self.Kp_db = 1                                                        # deadbeat gains
        self.Kd_db = self. T_DSP + (1/self.lam) * coth(self.lam * self.T_SSP) # deadbeat gains      
        self.sigma_P1 = self.lam * coth(0.5 * self.lam * self.T_SSP)          # orbital slope (P1)
        self.sigma_P2 = self.lam * tanh(0.5 * self.lam * self.T_SSP)          # orbital slope (P2)
        self.u_ff_x = 0.0
        self.u_ff_y = 0.0
        self.v_des_x = 0.0
        self.v_des_y = 0.0
        self.v_max = 0.3

        # period 2 feedforward foot placements
        self.u_L_bias = 0.28   # left is swing foot, add this to the feedforward foot placement
        self.u_R_bias = -0.28  # right is swing foot, add this to the feedforward foot placement

        # blending foot placement
        self.alpha = 1.0
        self.u_applied = 0
        self.bez_order = 7
        self.switched_stance_foot = False

        # timing variables
        self.t_current = 0
        self.t_phase = 0

        # instantiate inverse kinematics solver
        self.ik = InverseKinematics(self.plant)
        self.q_ik_sol = np.zeros(self.plant.num_positions())

        # inverse kinematics solver settings
        self.epsilon_feet = 0.00     # foot position tolerance     [m]
        self.epsilon_base = 0.00     # torso position tolerance    [m]
        self.foot_epsilon_orient = 0.0   # foot orientation tolerance  [deg]
        self.base_epsilon_orient = 0.0   # torso orientation tolerance [deg]
        self.tol_base = np.array([[self.epsilon_base], [self.epsilon_base], [self.epsilon_base]])
        self.tol_feet = np.array([[self.epsilon_feet], [self.epsilon_feet], [self.epsilon_feet]])

        # Add com position constraint (fixed constraint)
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
        
        # Add foot orientation constraints, aligns toe directions (remove and create this constraint at every discrete S2S step)
        self.r_com_cons = self.ik.AddOrientationConstraint(self.static_com_frame, RotationMatrix(), 
                                                           self.plant.world_frame(), RotationMatrix(), 
                                                           0.0)
        self.r_left_cons = self.ik.AddAngleBetweenVectorsConstraint(self.left_foot_frame, [1, 0, 0],
                                                                    self.plant.world_frame(), [1, 0, 0],
                                                                    0, 0)
        self.r_right_cons = self.ik.AddAngleBetweenVectorsConstraint(self.right_foot_frame, [1, 0, 0],
                                                                    self.plant.world_frame(), [1, 0, 0],
                                                                    0, 0)

    ########################################################################################################

    def draw_stance_frame(self, R, p):
        rpy = RotationMatrix(R).ToRollPitchYaw()
        
        # Basis vectors
        x_hat = R @ np.array([0.2, 0, 0]).reshape(3,1)
        y_hat = R @ np.array([0, 0.2, 0]).reshape(3,1)
        z_hat = R @ np.array([0, 0, 0.2]).reshape(3,1)
        
        px = x_hat * 0.5
        py = y_hat * 0.5
        pz = z_hat * 0.5

        Rx = RotationMatrix(R) @ RollPitchYaw(0, np.pi/2, 0).ToRotationMatrix()
        Ry = RotationMatrix(R) @ RollPitchYaw(np.pi/2, 0, 0).ToRotationMatrix()
        Rz = RotationMatrix(R) @ RollPitchYaw(0, 0, np.pi/2).ToRotationMatrix()
        self.meshcat.SetTransform("stance_x", RigidTransform(Rx, p + px), self.t_current)
        self.meshcat.SetTransform("stance_y", RigidTransform(Ry, p + py), self.t_current)
        self.meshcat.SetTransform("stance_z", RigidTransform(Rz, p + pz), self.t_current)


    # ---------------------------------------------------------------------------------------- #
    # update which foot should be swing and which one is stance
    def update_foot_role(self):
        
        # check if entered new step period
        if self.t_phase >= self.T_SSP:
            self.num_steps += 1
            self.update_foot_role_flag = True

        # update the foot role
        if self.update_foot_role_flag == True:

            # for the blended foot placement (to reset my foot placement low-pass filter)
            self.switched_stance_foot = True
            
            # left foot is swing foot
            if self.num_steps %2 == 0:

                # set the last known swing foot position as the desried stance foot position
                if self.num_steps == 0:
                    p_stance = self.plant.CalcPointsPositions(self.plant_context,
                                                              self.right_foot_frame, [0,0,0],
                                                              self.plant.world_frame())
                    R_stance = self.plant.CalcRelativeRotationMatrix(self.plant_context,
                                                                     self.plant.world_frame(),
                                                                     self.right_foot_frame)
                    self.p_swing_init_W = self.plant.CalcPointsPositions(self.plant_context,
                                                                        self.left_foot_frame, [0,0,0],
                                                                        self.plant.world_frame())
                else:
                    p_stance = self.plant.CalcPointsPositions(self.plant_context,
                                                            self.swing_foot_frame, [0,0,0],
                                                            self.plant.world_frame())
                    R_stance = self.plant.CalcRelativeRotationMatrix(self.plant_context,
                                                                    self.plant.world_frame(),
                                                                    self.swing_foot_frame)
                    # set the initial swing foot position
                    self.p_swing_init_W = self.plant.CalcPointsPositions(self.plant_context,
                                                                        self.stance_foot_frame, [0,0,0],
                                                                        self.plant.world_frame())
                self.control_stance_yaw = RollPitchYaw(R_stance).yaw_angle()
                self.p_control_stance_W = np.array([p_stance[0], p_stance[1], [self.z_offset]])
                self.R_control_stance_W = RotationMatrix(RollPitchYaw(0, 0, self.control_stance_yaw)).matrix()

                # remove the old orientation constraints
                self.ik.prog().RemoveConstraint(self.r_left_cons)
                self.ik.prog().RemoveConstraint(self.r_right_cons)
                self.ik.prog().RemoveConstraint(self.r_com_cons)

                # add the new orientation constraints
                self.r_com_cons = self.ik.AddOrientationConstraint(self.static_com_frame, RotationMatrix(),
                                                                   self.plant.world_frame(), RotationMatrix(self.R_control_stance_W),
                                                                   0.0)
                self.r_left_cons = self.ik.AddAngleBetweenVectorsConstraint(self.left_foot_frame, [1, 0, 0],
                                                                            self.plant.world_frame(), self.R_control_stance_W @ [1, 0, 0],
                                                                            0, self.foot_epsilon_orient * np.pi / 180)
                self.r_right_cons = self.ik.AddAngleBetweenVectorsConstraint(self.right_foot_frame, [1, 0, 0],
                                                                             self.plant.world_frame(), self.R_control_stance_W @ [1, 0, 0],
                                                                             0, self.foot_epsilon_orient * np.pi / 180)                

                # update the stance foot position constraints
                p_right_lb = self.p_control_stance_W - self.tol_feet
                p_right_ub = self.p_control_stance_W + self.tol_feet
                self.p_right_cons.evaluator().UpdateLowerBound(p_right_lb)
                self.p_right_cons.evaluator().UpdateUpperBound(p_right_ub)
                
                # switch the roles of the feet
                self.stance_foot_frame = self.right_foot_frame
                self.swing_foot_frame = self.left_foot_frame

                # draw the stance foot frame in meshcat
            
            # right foot is swing foot
            else:

                # set the last known swing foot position as the desried stance foot position
                if self.num_steps == 0:
                    p_stance = self.plant.CalcPointsPositions(self.plant_context,
                                                              self.left_foot_frame, [0,0,0],
                                                              self.plant.world_frame())
                    R_stance = self.plant.CalcRelativeRotationMatrix(self.plant_context,
                                                                     self.plant.world_frame(),
                                                                     self.left_foot_frame)
                    self.p_swing_init_W = self.plant.CalcPointsPositions(self.plant_context,
                                                                        self.right_foot_frame, [0,0,0],
                                                                        self.plant.world_frame())
                else:
                    p_stance = self.plant.CalcPointsPositions(self.plant_context,
                                                            self.swing_foot_frame,[0,0,0],
                                                            self.plant.world_frame())
                    R_stance = self.plant.CalcRelativeRotationMatrix(self.plant_context,
                                                                    self.plant.world_frame(),
                                                                    self.swing_foot_frame)
                    # set the initial swing foot position
                    self.p_swing_init_W = self.plant.CalcPointsPositions(self.plant_context,
                                                                        self.stance_foot_frame, [0,0,0],
                                                                        self.plant.world_frame())
                self.control_stance_yaw = RollPitchYaw(R_stance).yaw_angle()
                self.p_control_stance_W = np.array([p_stance[0], p_stance[1], [self.z_offset]])
                self.R_control_stance_W = RotationMatrix(RollPitchYaw(0, 0, self.control_stance_yaw)).matrix()

                # remove the old orientation constraints
                self.ik.prog().RemoveConstraint(self.r_left_cons)
                self.ik.prog().RemoveConstraint(self.r_right_cons)
                self.ik.prog().RemoveConstraint(self.r_com_cons)

                # add the new orientation constraints
                self.r_com_cons = self.ik.AddOrientationConstraint(self.static_com_frame, RotationMatrix(),
                                                                   self.plant.world_frame(), RotationMatrix(self.R_control_stance_W),
                                                                   0.0)
                self.r_left_cons = self.ik.AddAngleBetweenVectorsConstraint(self.left_foot_frame, [1, 0, 0],
                                                                            self.plant.world_frame(), self.R_control_stance_W @ [1, 0, 0],
                                                                            0, self.foot_epsilon_orient * np.pi / 180)
                self.r_right_cons = self.ik.AddAngleBetweenVectorsConstraint(self.right_foot_frame, [1, 0, 0],
                                                                            self.plant.world_frame(), self.R_control_stance_W @ [1, 0, 0],
                                                                            0, self.foot_epsilon_orient * np.pi / 180)

                # update the stance foot position constraints
                p_left_lb = self.p_control_stance_W - self.tol_feet
                p_left_ub = self.p_control_stance_W + self.tol_feet
                self.p_left_cons.evaluator().UpdateLowerBound(p_left_lb)
                self.p_left_cons.evaluator().UpdateUpperBound(p_left_ub)
                
                # switch the roles of the feet
                self.stance_foot_frame = self.left_foot_frame
                self.swing_foot_frame = self.right_foot_frame

            # reset the foot role flag after switching
            self.update_foot_role_flag = False  

        # update the phase time
        self.t_phase = self.t_current - self.num_steps * self.T_SSP

    # ---------------------------------------------------------------------------------------- #
    
    # update the COM state of the HLIP (need this incase you want to change v_des)
    def update_hlip_state_H(self):
        
        # x-direction [p, v] in local stance foot frame (P1 Orbit)
        # Eq (20) Xiaobing Xiong, Ames
        T = self.T_SSP + self.T_DSP
        self.p_H_minus_x = (self.v_des_x * T) / (2 + self.T_DSP * self.sigma_P1)
        self.v_H_minus_x = self.sigma_P1 * (self.v_des_x * T) / (2 + self.T_DSP * self.sigma_P1)

        # y-direction [p, v] in local stance foot frame (P2 Orbit), must satisfy u_L + u_R = 2 * v_des * T
        # Eq (21) Xiaobing Xiong, Ames
        self.u_L = self.u_L_bias + self.v_des_y * (self.T_SSP + self.T_DSP)
        self.u_R = self.u_R_bias + self.v_des_y * (self.T_SSP + self.T_DSP)
        if self.swing_foot_frame == self.left_foot_frame:
            u_star = self.u_L    
        elif self.swing_foot_frame == self.right_foot_frame:
            u_star = self.u_R  
        self.d2 = self.lam**2 * (self.sech(0.5 * self.lam * self.T_SSP))**2 * (T * self.v_des_y) / (self.lam**2 * self.T_DSP + 2 * self.sigma_P2)
        self.p_H_minus_y = (u_star - self.T_DSP * self.d2) / (2 + self.T_DSP * self.sigma_P2)
        self.v_H_minus_y = self.sigma_P2 * self.p_H_minus_y + self.d2

    # update the COM state of the Robot
    def update_hlip_state_R(self):

        # compute the dynamic p_com in world frame
        p_com_W = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context).reshape(3,1)
        p_com_control_stance = self.R_control_stance_W.T @ (p_com_W - self.p_control_stance_W)

        # compute v_com, via Jacobian (3xn) and generalized velocity (nx1)
        J = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(self.plant_context, 
                                                                     JacobianWrtVariable.kV,
                                                                     self.plant.world_frame(), 
                                                                     self.plant.world_frame())
        v = self.plant.GetVelocities(self.plant_context)
        v_com_W = (J @ v).reshape(3,1)
        v_com_control_stance = self.R_control_stance_W.T @ v_com_W

        # update HLIP state for the robot
        self.p_R = p_com_control_stance
        self.v_R = v_com_control_stance

        # compute the preimpact state
        x0_x = np.array([self.p_R[0][0], self.v_R[0][0]]).T
        x0_y = np.array([self.p_R[1][0], self.v_R[1][0]]).T
        x_R_minus_x = (sp.linalg.expm(self.A * (self.T_SSP - self.t_phase)) @ x0_x).reshape(2,1)
        x_R_minus_y = (sp.linalg.expm(self.A * (self.T_SSP - self.t_phase)) @ x0_y).reshape(2,1)
        self.p_R_minus_x = x_R_minus_x[0][0]
        self.v_R_minus_x = x_R_minus_x[1][0]
        self.p_R_minus_y = x_R_minus_y[0][0]
        self.v_R_minus_y = x_R_minus_y[1][0]

        # visualize the CoM position
        self.meshcat.SetTransform("com_pos", RigidTransform(p_com_W), self.t_current)

        # visualize the CoM velocity
        p_com_xy = np.array([p_com_W[0], p_com_W[1]])
        v_com_xy = np.array([v_com_W[0], v_com_W[1]])
        yaw = np.arctan2(v_com_W[1], v_com_W[0])
        v_norm = np.linalg.norm(v_com_xy)

        # setting the rigid transform for the CoM velocity
        if v_norm <= 0.001:
            cyl = Cylinder(0.005, 0.001)
            self.meshcat.SetObject("com_vel",cyl, self.green_color)
            # self.meshcat.SetProperty("com_vel", "scale", [1, 1, 1000], self.t_current)
            self.meshcat.SetTransform("com_vel", RigidTransform(RotationMatrix(),p_com_W), self.t_current)
        else:
            cyl = Cylinder(0.005, v_norm)
            self.meshcat.SetObject("com_vel",cyl, self.green_color)
            rpy = RollPitchYaw(0.0, np.pi/2, yaw)
            p = 0.5 * (v_com_xy + 2*p_com_xy)
            p = np.array([p[0], p[1], p_com_W[2]])
            self.meshcat.SetTransform("com_vel", RigidTransform(rpy,p), self.t_current)

        # draw teh stance foot control frame
        self.draw_stance_frame(self.R_control_stance_W, self.p_control_stance_W)

    # ---------------------------------------------------------------------------------------- #
    # update where to place the foot, (i.e., apply discrete control to the HLIP model)
    def update_foot_placement(self):

        # x-direction [p, v] in world frame
        # px = self.p_R[0][0]
        # vx = self.v_R[0][0]
        px_R_minus = self.p_R_minus_x
        vx_R_minus = self.v_R_minus_x
        px_H_minus = self.p_H_minus_x
        vx_H_minus = self.v_H_minus_x

        # y-direction [p, v] in world frame
        # py = self.p_R[1][0]
        # vy = self.v_R[1][0]
        py_R_minus = self.p_R_minus_y
        vy_R_minus = self.v_R_minus_y
        py_H_minus = self.p_H_minus_y
        vy_H_minus = self.v_H_minus_y

        # TODO: there's seems to be a flipped sign on the y-direction that causes bad tracking. 

        # compute foot placement in x-direction (local stance foot frame)
        # u_x = self.u_ff_x + self.v_des_x * self.T_SSP + self.Kp_db * (px - 0) + self.Kd_db * (vx - 0)
        # u_x = self.u_ff_x + self.v_des_x * self.T_SSP + self.Kp_db * (px - px_H_minus) + self.Kd_db * (vx - vx_H_minus)
        u_x = self.u_ff_x + self.v_des_x * self.T_SSP + self.Kp_db * (px_R_minus - px_H_minus) + self.Kd_db * (vx_R_minus - vx_H_minus)

        # compute foot placement in y-direction (local stance foot frame)
        # u_y = self.u_ff_y + self.v_des_y * self.T_SSP + self.Kp_db * (py - 0) + self.Kd_db * (vy - 0)
        # u_y = self.u_ff_y + self.v_des_y * self.T_SSP + self.Kp_db * (py - py_H_minus) + self.Kd_db * (vy - vy_H_minus)
        u_y = self.u_ff_y + self.v_des_y * self.T_SSP + self.Kp_db * (py_R_minus - py_H_minus) + self.Kd_db * (vy_R_minus - vy_H_minus)

        if self.swing_foot_frame == self.left_foot_frame:
            u_y += self.u_L
        elif self.swing_foot_frame == self.right_foot_frame:
            u_y += self.u_R

        # check if in new step period
        # if self.switched_stance_foot == True:
        #     # reset the filter (blending history)
        #     self.u_applied_x = u_x
        #     self.u_applied_y = u_y
        #     self.switched_stance_foot = False
        # else:
        #     # compute the blended foot placement
        #     self.u_applied_x = self.alpha * u_x + (1 - self.alpha) * self.u_applied_x
        #     self.u_applied_y = self.alpha * u_y + (1 - self.alpha) * self.u_applied_y

        self.u_applied_x = u_x
        self.u_applied_y = u_y

        return self.u_applied_x, self.u_applied_y

    # ---------------------------------------------------------------------------------------- #
    # update the desired foot trajectories
    def update_foot_traj(self):

        # swing foot targets in world frame
        u_app_x, u_app_y = self.update_foot_placement()   # foot placement (in stance foot control frame)
        u0_x = self.p_swing_init_W[0][0]                  # inital swing foot position in world frame
        u0_y = self.p_swing_init_W[1][0]                  # inital swing foot position in world frame
        uf_x = self.p_control_stance_W[0][0] + u_app_x * np.cos(self.control_stance_yaw) - u_app_y * np.sin(self.control_stance_yaw) # final swing foot position in world frame 
        uf_y = self.p_control_stance_W[1][0] + u_app_x * np.sin(self.control_stance_yaw) + u_app_y * np.cos(self.control_stance_yaw) # final swing foot position in world frame

        # compute primary bezier curve control points
        if self.bez_order == 7:
            ctrl_pts_x = np.array([[u0_x],[u0_x],[u0_x],[(u0_x+uf_x)/2],[uf_x],[uf_x],[uf_x]])
            ctrl_pts_y = np.array([[u0_y],[u0_y],[u0_y],[(u0_y+uf_y)/2],[uf_y],[uf_y],[uf_y]])
            ctrl_pts_z = np.array([[self.z0],[self.z0],[self.z0],[(16/5)*self.z_apex],[self.zf],[self.zf],[self.zf]]) + self.z_offset

        elif self.bez_order == 5:
            ctrl_pts_x = np.array([[u0_x],[u0_x],[(u0_x+uf_x)/2],[uf_x],[uf_x]])
            ctrl_pts_y = np.array([[u0_y],[u0_y],[(u0_y+uf_y)/2],[uf_y],[uf_y]])
            ctrl_pts_z = np.array([[self.z0],[self.z0],[(8/3)*self.z_apex],[self.zf],[self.zf]]) + self.z_offset
   
        # set the primary control points
        ctrl_pts = np.vstack((ctrl_pts_x.T,
                              ctrl_pts_y.T,
                              ctrl_pts_z.T))

        # evaluate bezier at time t
        bezier = BezierCurve(0, self.T_SSP, ctrl_pts)
        b = np.array(bezier.value(self.t_phase))

        # swing and stance foot targets
        primary_swing_target = np.array([b.T[0][0], 
                                         b.T[0][1], 
                                         b.T[0][2]])[None].T
        
        # # linearly interpolate the current swnig foot position with the primary target
        # alpha = self.t_phase / self.T_SSP
        # p_swing_current = self.plant.CalcPointsPositions(self.plant_context,
        #                                                  self.swing_foot_frame,
        #                                                  [0,0,0],
        #                                                  self.plant.world_frame())
        # swing_target_W = (1 - alpha) * p_swing_current + (alpha) * primary_swing_target
        swing_target_W = primary_swing_target

        return swing_target_W  # return the desired swing foot in world frame
    
    # ---------------------------------------------------------------------------------------- #
    # given desired foot and torso positions, solve the IK problem
    def DoInverseKinematics(self, p_swing_W):

        # update the torso position constraints
        p_static_com_current_W = self.plant.CalcPointsPositions(self.plant_context,
                                                               self.static_com_frame,
                                                               [0,0,0],
                                                               self.plant.world_frame())
        p_static_com_target = np.array([p_static_com_current_W[0], p_static_com_current_W[1], [self.z_nom]])
        self.p_com_cons.evaluator().UpdateLowerBound(p_static_com_target - self.tol_base)
        self.p_com_cons.evaluator().UpdateUpperBound(p_static_com_target + self.tol_base)

        # update the foot position constraints
        if self.swing_foot_frame == self.left_foot_frame:
            p_left_lb = p_swing_W - self.tol_feet
            p_left_ub = p_swing_W + self.tol_feet
            self.p_left_cons.evaluator().UpdateLowerBound(p_left_lb)
            self.p_left_cons.evaluator().UpdateUpperBound(p_left_ub)

            # update teh foot target visuals
            self.meshcat.SetTransform("p_left", RigidTransform(p_swing_W), self.t_current)
            self.meshcat.SetTransform("p_right", RigidTransform(self.p_control_stance_W), self.t_current)

        # update the foot position constraints
        elif self.swing_foot_frame == self.right_foot_frame:
            p_right_lb = p_swing_W - self.tol_feet
            p_right_ub = p_swing_W + self.tol_feet
            self.p_right_cons.evaluator().UpdateLowerBound(p_right_lb)
            self.p_right_cons.evaluator().UpdateUpperBound(p_right_ub)

            # update teh foot target visuals
            self.meshcat.SetTransform("p_right", RigidTransform(p_swing_W), self.t_current)
            self.meshcat.SetTransform("p_left", RigidTransform(self.p_control_stance_W), self.t_current)

        # solve the IK problem        
        initial_guess = self.plant.GetPositions(self.plant_context)
        self.ik.prog().SetInitialGuess(self.ik.q(), initial_guess)
        res = SnoptSolver().Solve(self.ik.prog())
        
        return res

    # ---------------------------------------------------------------------------------------- #
    def CalcOutput(self, context, output):

        print("\n *************************************** \n")
        print("time: ", self.t_current) 

        # set our interal model to match the state estimate
        x_hat = self.EvalVectorInput(context, 0).get_value()
        self.plant.SetPositionsAndVelocities(self.plant_context, x_hat)
        self.t_current = context.get_time()

        # evaluate the joystick command
        joy_command = self.gamepad_port.Eval(context)
        self.v_des_x = joy_command[1] * self.v_max
        self.v_des_y = joy_command[0] * self.v_max

        # update everything
        self.update_foot_role()
        self.update_hlip_state_H()
        self.update_hlip_state_R()

        # compute desired swing foot trajectory
        p_swing_W = self.update_foot_traj()

        # solve the IK problem
        res = self.DoInverseKinematics(p_swing_W)

        # extract the IK solution
        if res.is_success():
            q_ik = res.GetSolution(self.ik.q())
            self.q_ik_sol = q_ik
        else:
            q_ik = self.plant.GetPositions(self.plant_context)
            print("\n ************* IK failed! ************* \n")

        # compute the nominal state
        q_des = np.array([q_ik[7],  q_ik[8],  q_ik[9],  q_ik[10], q_ik[11],  # left leg:  hip_yaw, hip_roll, hip_pitch, knee, ankle 
                          q_ik[12], q_ik[13], q_ik[14], q_ik[15], q_ik[16]]) # right leg: hip_yaw, hip_roll, hip_pitch, knee, ankle
        # q_des = np.array([0, 0, 0, 0, 0, 
        #                   0, 0, 0, 0, 0])
        v_des = np.zeros(self.plant.num_actuators())
        x_des = np.block([q_des, v_des])

        output.SetFromVector(x_des)

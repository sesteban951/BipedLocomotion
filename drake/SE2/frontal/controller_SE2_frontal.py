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

        self.meshcat.SetObject("com_pos", self.sphere, self.green_color)
        self.meshcat.SetObject("p_right", self.sphere, self.red_color)
        self.meshcat.SetObject("p_left", self.sphere, self.blue_color)

        # create internal model of the robot
        self.plant = MultibodyPlant(0)
        Parser(self.plant).AddModels(model_file)
        self.plant.Finalize()
        self.plant_context = self.plant.CreateDefaultContext()

        # Leaf System input port
        self.input_port = self.DeclareVectorInputPort("x_hat", 
                                                      BasicVector(self.plant.num_positions() + self.plant.num_velocities()))
        self.gamepad_port = self.DeclareVectorInputPort("joy_command",
                                                        BasicVector(5))  # LS_x, LS_y, RS_x, A button (Xbox), RT
        self.DeclareVectorOutputPort("x_des",
                                     BasicVector(2 * self.plant.num_actuators()),
                                     self.CalcOutput)
    
        # relevant frames
        self.static_com_frame = self.plant.GetFrameByName("static_com") # nominal is 0.734 z in world frame
        self.left_foot_frame = self.plant.GetFrameByName("left_foot")
        self.right_foot_frame = self.plant.GetFrameByName("right_foot")
        self.stance_foot_frame = self.right_foot_frame
        self.swing_foot_frame  = self.left_foot_frame

        # Foot position Variables
        self.p_stance = np.zeros(3)
        self.p_swing_init = np.zeros(3)

        # for updating number of steps amd switching foot stance/swing
        self.update_foot_role_flag = True
        self.num_steps = 0

        # walking parameters
        self.z_nom = 0.64
        self.T_SSP = 0.3   # single support phase
        self.T_DSP = 0.0   # double support phase

        # HLIP parameters
        g = 9.81
        self.lam = np.sqrt(g/self.z_nom)
        self.A = np.array([[0,           1],   # LIP drift matrix
                           [self.lam**2, 0]])

        # Robot LIP state
        self.p_com = np.zeros(3)    # center of mass state in world frame
        self.v_com = np.zeros(3)    # center of mass velocity in world frame
        self.p_R = np.zeros(3)      # center of mass state in stance foot frame
        self.v_R = np.zeros(3)      # center of mass velocity in stance foot frame

        # LIP model preimpact states
        self.p_H_minus = 0
        self.v_H_minus = 0
        self.p_R_minus = 0
        self.v_R_minus = 0

        # y-direction foot palcement offsets
        self.u_L_bias =  0.28   # left is swing foot, add this to feedforawrd footplacement term
        self.u_R_bias = -0.28   # right is swing foot, add this to feedforawrd footplacement term

        # swing foot parameters
        self.z_apex = 0.08    # NOTE: this is affected by bezier curve swing belnding
        self.z_offset = 0.0
        self.z0 = 0.0
        self.zf = 0.0

        # create lambda function for hyperbolic trig
        coth = lambda x: (np.exp(2 * x) + 1) / (np.exp(2 * x) - 1)
        tanh = lambda x: (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)
        self.sech = lambda x: 1 / np.cosh(x)

        # define the deadbeat gains and orbital slopes
        self.Kp_db = 1                                                        # deadbeat gains
        self.Kd_db = self. T_DSP + (1/self.lam) * coth(self.lam * self.T_SSP) # deadbeat gains      
        self.sigma_P2 = self.lam * tanh(0.5 * self.lam * self.T_SSP)          # orbital slope (P2)
        self.v_des = 0.0
        self.v_max = 0.3

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
        epsilon_feet = 0.00     # foot position tolerance     [m]
        epsilon_base = 0.00     # torso position tolerance    [m]
        foot_epsilon_orient = 0.   # foot orientation tolerance  [deg]
        base_epsilon_orient = 0.   # torso orientation tolerance [deg]
        self.tol_base = np.array([[np.inf], [epsilon_base], [epsilon_base]])  # y-z only
        self.tol_feet = np.array([[epsilon_feet], [epsilon_feet], [epsilon_feet]])  # y-z only

        # Add com position constraint
        self.p_com_cons = self.ik.AddPositionConstraint(self.static_com_frame, [0, 0, 0], 
                                                        self.plant.world_frame(), 
                                                        [0, 0, 0], [0, 0, 0]) 
        
        # Add com orientation constraint
        self.r_com_cons = self.ik.AddOrientationConstraint(self.static_com_frame, RotationMatrix(),
                                                           self.plant.world_frame(), RotationMatrix(),
                                                           base_epsilon_orient * (np.pi/180))
        
        # Add foot position constraints
        self.p_left_cons =  self.ik.AddPositionConstraint(self.left_foot_frame, [0, 0, 0],
                                                          self.plant.world_frame(), 
                                                          [0, 0, 0], [0, 0, 0])
        self.p_right_cons = self.ik.AddPositionConstraint(self.right_foot_frame, [0, 0, 0],
                                                          self.plant.world_frame(), 
                                                          [0, 0, 0], [0, 0, 0]) 
        
        # Add foot orientation constraints
        self.r_left_cons =  self.ik.AddOrientationConstraint(self.left_foot_frame, RotationMatrix(),
                                                             self.plant.world_frame(), RotationMatrix(),
                                                             foot_epsilon_orient * (np.pi/180))
        self.r_right_cons = self.ik.AddOrientationConstraint(self.right_foot_frame, RotationMatrix(),
                                                             self.plant.world_frame(), RotationMatrix(),
                                                             foot_epsilon_orient * (np.pi/180))

    ########################################################################################################

    # ---------------------------------------------------------------------------------------- #
    # update which foot should be swing and which one is stance
    def update_foot_role(self):
        
        # check if entered new step period
        if self.t_phase >= self.T_SSP:
            self.num_steps += 1
            self.update_foot_role_flag = True

        # update the foot role
        if self.update_foot_role_flag == True:

            # for the blended foot placement
            self.switched_stance_foot = True
            
            # left foot is swing foot
            if self.num_steps %2 == 0:

                # set the last known swing foot position as the desried stance foot position
                self.p_stance = self.plant.CalcPointsPositions(self.plant_context,
                                                               self.swing_foot_frame,
                                                               [0,0,0],
                                                               self.plant.world_frame())
                # set the initial swing foot position
                self.p_swing_init = self.plant.CalcPointsPositions(self.plant_context,
                                                                      self.stance_foot_frame,
                                                                      [0,0,0],
                                                                      self.plant.world_frame())
                # switch the roles of the feet
                self.stance_foot_frame = self.right_foot_frame
                self.swing_foot_frame = self.left_foot_frame
    
            # right foot is swing foot
            else:

                # set the last known swing foot position as the desried stance foot position
                self.p_stance = self.plant.CalcPointsPositions(self.plant_context,
                                                               self.swing_foot_frame,
                                                               [0,0,0],
                                                               self.plant.world_frame())
                # set the initial swing foot position
                self.p_swing_init = self.plant.CalcPointsPositions(self.plant_context,
                                                                      self.stance_foot_frame,
                                                                      [0,0,0],
                                                                      self.plant.world_frame())
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

        # y-direction [p, v] in local stance foot frame (P2 Orbit), must satisfy u_L + u_R = 2 * v_des * T
        # Eq (21) Xiaobing Xiong, Ames
        T = self.T_SSP + self.T_DSP
        self.u_L = self.u_L_bias + self.v_des * (self.T_SSP + self.T_DSP)
        self.u_R = self.u_R_bias + self.v_des * (self.T_SSP + self.T_DSP)
        if self.swing_foot_frame == self.left_foot_frame:
            u_star = self.u_L    
        elif self.swing_foot_frame == self.right_foot_frame:
            u_star = self.u_R  
        self.d2 = self.lam**2 * (self.sech(0.5 * self.lam * self.T_SSP))**2 * (T * self.v_des) / (self.lam**2 * self.T_DSP + 2 * self.sigma_P2)
        self.p_H_minus_y = (u_star - self.T_DSP * self.d2) / (2 + self.T_DSP * self.sigma_P2)
        self.v_H_minus_y = self.sigma_P2 * self.p_H_minus_y + self.d2

    # update the COM state of the Robot
    def update_hlip_state_R(self):
        
        # compute the static p_com in world frame
        self.p_static_com = self.plant.CalcPointsPositions(self.plant_context,
                                                           self.static_com_frame,
                                                           [0,0,0],
                                                           self.plant.world_frame()).flatten()

        # compute the dynamic p_com in world frame
        self.p_com = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context)

        # compute v_com, via Jacobian (3xn) and generalized velocity (nx1)
        J = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(self.plant_context, 
                                                                     JacobianWrtVariable.kV,
                                                                     self.plant.world_frame(), 
                                                                     self.plant.world_frame())
        v = self.plant.GetVelocities(self.plant_context)
        v_all = J @ v
        self.v_com = v_all[0:3]

        # update CoM info
        self.v_com = self.v_com.T
        self.p_com = self.p_com.T

        # update HLIP state for the robot
        self.p_R = (self.p_com - self.p_stance.T).T
        self.v_R = self.v_com

        # compute the preimpact state
        y0 = np.array([self.p_R[1][0], self.v_R[1]]).T
        x_R_minus = sp.linalg.expm(self.A * (self.T_SSP - self.t_phase)) @ y0
        self.p_R_minus = x_R_minus[0]
        self.v_R_minus = x_R_minus[1]

        # visualize the CoM position
        self.meshcat.SetTransform("com_pos", RigidTransform(self.p_com), self.t_current)

        # visualize the CoM velocity
        if self.v_com[1] >= 0:
            v_norm = np.linalg.norm(self.v_com[1]) + 1E-5
            cyl = Cylinder(0.005, v_norm)
            self.meshcat.SetObject("com_vel",cyl, self.green_color)
            rpy = RollPitchYaw(np.pi/2, 0, 0)
            p = self.p_com + np.array([v_norm,0,0]) * 0.5
            self.meshcat.SetTransform("com_vel", RigidTransform(rpy,p), self.t_current)
        else:
            v_norm = np.linalg.norm(self.v_com[1]) + 1E-5
            cyl = Cylinder(0.005, v_norm)
            self.meshcat.SetObject("com_vel",cyl, self.green_color)
            rpy = RollPitchYaw(-np.pi/2, 0, 0)
            p = self.p_com + np.array([-v_norm,0,0]) * 0.5
            self.meshcat.SetTransform("com_vel", RigidTransform(rpy,p), self.t_current)

    # ---------------------------------------------------------------------------------------- #
    # update where to place the foot, (i.e., apply discrete control to the HLIP model)
    def update_foot_placement(self):

        # x-direction [p, v]
        px = self.p_R[1]
        vx = self.v_R[1]
        px_R_minus = self.p_R_minus
        vx_R_minus = self.v_R_minus
        px_H_minus = self.p_H_minus
        vx_H_minus = self.v_H_minus

        # compute foot placement
        u = self.v_des * self.T_SSP + self.Kp_db * (px_R_minus - px_H_minus) + self.Kd_db * (vx_R_minus - vx_H_minus)    # HLIP, preimpact

        if self.swing_foot_frame == self.left_foot_frame:
            u += self.u_L
        elif self.swing_foot_frame == self.right_foot_frame:
            u += self.u_R

        # # check if in new step period
        # if self.switched_stance_foot == True:
        #     # reset the filter (blending history)
        #     self.u_applied = u
        #     self.switched_stance_foot = False
        # else:
        #     # compute the blended foot placement
        #     self.u_applied = self.alpha * u + (1 - self.alpha) * self.u_applied

        return u

    # ---------------------------------------------------------------------------------------- #
    
    # update the desired foot trjectories
    def update_foot_traj(self):

        # swing foot traget
        u0 = self.p_swing_init[1][0]  # inital swing foot position
        uf = self.p_stance[1][0] + self.update_foot_placement()

        # compute primary bezier curve control points
        if self.bez_order == 7:
            ctrl_pts_y = np.array([[u0],[u0],[u0],[(u0+uf)/2],[uf],[uf],[uf]])
            ctrl_pts_z = np.array([[self.z0],[self.z0],[self.z0],[(16/5)*self.z_apex],[self.zf],[self.zf],[self.zf]]) + self.z_offset

        elif self.bez_order == 5:
            ctrl_pts_y = np.array([[u0],[u0],[(u0+uf)/2],[uf],[uf]])
            ctrl_pts_z = np.array([[self.z0],[self.z0],[(8/3)*self.z_apex],[self.zf],[self.zf]]) + self.z_offset
   
        # set the primary control points
        ctrl_pts = np.vstack((ctrl_pts_y.T, 
                              ctrl_pts_z.T))
        
        # evaluate bezier at time t
        bezier = BezierCurve(0, self.T_SSP, ctrl_pts)
        b = np.array(bezier.value(self.t_phase))

        # swing ans stance foot targets
        primary_swing_target = np.array([0.0, 
                                         b.T[0][0], 
                                         b.T[0][1]])[None].T
        stance_target = np.array([0.0, 
                                  self.p_stance[1][0], 
                                  self.z_offset])[None].T
        
        # linearly interpolate the current swnig foot position with the primary target
        # alpha = self.t_phase / self.T_SSP
        # p_swing_current = self.plant.CalcPointsPositions(self.plant_context,
        #                                                  self.swing_foot_frame,
        #                                                  [0,0,0],
        #                                                  self.plant.world_frame())
        # swing_target = (1 - alpha) * p_swing_current + (alpha) * primary_swing_target
        swing_target = primary_swing_target

        # left foot in swing
        if self.num_steps % 2 == 0:
            p_right = stance_target
            p_left = swing_target
        # right foot in swing
        else:
            p_right = swing_target
            p_left = stance_target

        self.meshcat.SetTransform("p_right", RigidTransform([0.0, p_right[1][0], p_right[2][0]]), self.t_current)        
        self.meshcat.SetTransform("p_left", RigidTransform([0.0, p_left[1][0], p_left[2][0]]), self.t_current)

        return p_right, p_left
    
    # ---------------------------------------------------------------------------------------- #
    # given desired foot and torso positions, solve the IK problem
    def DoInverseKinematics(self, p_right, p_left):

        # update the torso position constraints
        p_static_com_W = self.plant.CalcPointsPositions(self.plant_context,
                                                        self.static_com_frame,
                                                        [0,0,0],
                                                        self.plant.world_frame())
        p_static_com_target = np.array([[0], p_static_com_W[1], [self.z_nom]])
        self.p_com_cons.evaluator().UpdateLowerBound(p_static_com_target - self.tol_base)
        self.p_com_cons.evaluator().UpdateUpperBound(p_static_com_target + self.tol_base)

        print("p_static_com_target: ", p_static_com_target)

        # Update constraints on the positions of the feet
        p_left_lb = p_left - self.tol_feet
        p_left_ub = p_left + self.tol_feet
        p_right_lb = p_right - self.tol_feet
        p_right_ub = p_right + self.tol_feet
        self.p_left_cons.evaluator().UpdateLowerBound(p_left_lb)
        self.p_left_cons.evaluator().UpdateUpperBound(p_left_ub)
        self.p_right_cons.evaluator().UpdateLowerBound(p_right_lb)
        self.p_right_cons.evaluator().UpdateUpperBound(p_right_ub)

        print("p_left: ", p_left)
        print("p_right: ", p_right)

        # solve the IK problem        
        initial_guess = self.plant.GetPositions(self.plant_context)
        # initial_guess = self.q_ik_sol
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
        self.v_des = joy_command[0] * self.v_max

        # update everything
        self.update_foot_role()
        self.update_hlip_state_H()
        self.update_hlip_state_R()

        # compute desired foot trajectories
        p_right, p_left = self.update_foot_traj()

        # solve the inverse kinematics problem
        p_right_des = np.array([[0], p_right[1], p_right[2]])
        p_left_des = np.array([[0], p_left[1], p_left[2]])

        # solve the IK problem
        res = self.DoInverseKinematics(p_right_des, 
                                       p_left_des)
        
        # extract the IK solution
        if res.is_success():
            q_ik = res.GetSolution(self.ik.q())
            self.q_ik_sol = q_ik
        else:
            q_ik = self.q_ik_sol
            print("\n ************* IK failed! ************* \n")

        # compute the nominal state
        q_des = np.array([q_ik[3], q_ik[4], q_ik[5], q_ik[6],    # left leg: hip_pitch, knee, ankle
                          q_ik[7], q_ik[8], q_ik[9], q_ik[10]]) # right leg: hip_pitch, knee, ankle
        # q_des = np.zeros(self.plant.num_actuators())
        v_des = np.zeros(self.plant.num_actuators())
        x_des = np.block([q_des, v_des])

        output.SetFromVector(x_des)



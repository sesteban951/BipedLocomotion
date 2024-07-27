#!/usr/bin/env python

from pydrake.all import *
import numpy as np

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

        # Leaf System input port
        self.input_port = self.DeclareVectorInputPort(
                            "x_hat", 
                            BasicVector(self.plant.num_positions() + self.plant.num_velocities()))
        self.DeclareVectorOutputPort(
                            "x_des",
                            BasicVector(2 * self.plant.num_actuators()),
                            self.CalcOutput)
    
        # relevant frames
        self.torso_frame = self.plant.GetFrameByName("torso")
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
        self.z_nom = 0.95
        self.T = 0.25

        # HLIP parameters
        g = 9.81
        lam = np.sqrt(g/self.z_nom)
        self.A = np.array([[0,      1],   # LIP drift matrix
                           [lam**2, 0]])

        # Robot LIP state
        self.p_com = np.zeros(3)
        self.v_com = np.zeros(3)
        self.p_R = np.zeros(3)
        self.v_R = np.zeros(3)
        self.p_com_B = np.zeros(3)

        # swing foot parameters
        self.z_apex = 0.15
        self.z_offset = 0.03
        self.z0 = 0.0
        self.zf = 0.0

        # Gains (using the deadbeat gain)
        coth = (np.exp(2 * self.T * lam) + 1) / (np.exp(2 * self.T * lam) - 1)
        self.Kp_db = 1
        self.Kd_db = (1/lam) * coth
        self.u_ff = 0.0
        self.v_des = 0.0
        self.com_x_offset = 0.0
        self.com_z_offset = 0.0

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

        # inverse kinematics solver settings
        self.epsilon_feet = 0.01   # foot position tolerance     [m]
        self.epsilon_base = 0.01   # torso position tolerance    [m]
        epsilon_orient = 1.0      # torso orientation tolerance [deg]
        bias_orient = 1.0         # torso orientation bias      [deg]
        self.tol_feet = np.array([[self.epsilon_feet], [np.inf], [self.epsilon_feet]])  # x-z only
        self.tol_base = np.array([[self.epsilon_base], [np.inf], [self.epsilon_base]])  # z only

        # Add torso position constraint
        self.p_torso_cons = self.ik.AddPositionConstraint(self.torso_frame, [0, 0, 0], 
                                                          self.plant.world_frame(), 
                                                          [0,0,0], [0,0,0]) 
        
        # Add torso orientation constraint (torso coord frame, x is z-world, y is neg y-world, z is x-world)
        R_bias = RotationMatrix(RollPitchYaw(0, bias_orient * (np.pi/180), 0))
        self.r_torso_cons = self.ik.AddOrientationConstraint(self.torso_frame, RotationMatrix(),
                                                            self.plant.world_frame(), R_bias,
                                                            epsilon_orient * (np.pi/180))
        
        # Add foot constraints
        self.p_left_cons =  self.ik.AddPositionConstraint(self.left_foot_frame, [0, 0, 0],
                                                          self.plant.world_frame(), 
                                                          [0,0,0], [0,0,0])
        self.p_right_cons = self.ik.AddPositionConstraint(self.right_foot_frame, [0, 0, 0],
                                                          self.plant.world_frame(), 
                                                          [0,0,0], [0,0,0]) 

        
    ########################################################################################################

    # ---------------------------------------------------------------------------------------- #
    # update which foot should be swing and which one is stance
    def update_foot_role(self):
        
        # check if entered new step period
        if self.t_phase >= self.T:
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
        self.t_phase = self.t_current - self.num_steps * self.T

    # ---------------------------------------------------------------------------------------- #
    # update the COM state of the Robot
    def update_hlip_state_R(self):
        
        # compute p_com
        self.p_com = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context)

        # get in the torso frame
        self.p_com_B = self.plant.CalcPointsPositions(self.plant_context,
                                                      self.plant.world_frame(),
                                                      self.p_com,
                                                      self.torso_frame).flatten()
        
        # apply the com offset in body frame
        self.p_com_B[0] = self.p_com_B[0] + self.com_x_offset # x offset
        
        # apply the com offset in world frame
        self.p_com = self.plant.CalcPointsPositions(self.plant_context,
                                                    self.torso_frame,
                                                    self.p_com_B,
                                                    self.plant.world_frame()).flatten()

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

    # ---------------------------------------------------------------------------------------- #
    # update where to place the foot, (i.e., apply discrete control to the HLIP model)
    def update_foot_placement(self):

        # x-direction [p, v]
        px = self.p_R[0]
        vx = self.v_R[0]

        # desired velocity
        v_des = self.v_des

        # compute raibert heuristic foot gains
        kp = self.Kp_db
        kd = self.Kd_db
        u = self.u_ff + kp * (px) + kd * (vx - v_des)

        # check if in new step period
        if self.switched_stance_foot == True:
            # reset the filter (blending history)
            self.u_applied = u[0]
            self.switched_stance_foot = False
        else:
            # compute the blended foot placement
            self.u_applied = self.alpha * u[0] + (1 - self.alpha) * self.u_applied

        return self.u_applied

    # ---------------------------------------------------------------------------------------- #
    # update the desired foot trjectories
    def update_foot_traj(self):

        # swing foot traget
        u0 = self.p_swing_init[0][0]  # inital swing foot position
        uf = self.p_stance[0][0] + self.update_foot_placement()

        # compute primary bezier curve control points
        if self.bez_order == 7:
            ctrl_pts_x = np.array([[u0],[u0],[u0],[(u0+uf)/2],[uf],[uf],[uf]])
            ctrl_pts_z = np.array([[self.z0],[self.z0],[self.z0],[(16/5)*self.z_apex],[self.zf],[self.zf],[self.zf]]) + self.z_offset

        elif self.bez_order == 5:
            ctrl_pts_x = np.array([[u0],[u0],[(u0+uf)/2],[uf],[uf]])
            ctrl_pts_z = np.array([[self.z0],[self.z0],[(8/3)*self.z_apex],[self.zf],[self.zf]]) + self.z_offset
   
        # set the primary control points
        ctrl_pts = np.vstack((ctrl_pts_x.T, 
                              ctrl_pts_z.T))

        # evaluate bezier at time t
        bezier = BezierCurve(0, self.T, ctrl_pts)
        b = np.array(bezier.value(self.t_phase))

        # swing ans stance foot targets
        primary_swing_target = np.array([b.T[0][0], 
                                         0.0, 
                                         b.T[0][1]])[None].T
        stance_target = np.array([self.p_stance[0][0], 
                                          0, 
                                          self.z_offset])[None].T
        
        # linearly interpolate the current swnig foot position with the primary target
        alpha = self.t_phase / self.T
        p_swing_current = self.plant.CalcPointsPositions(self.plant_context,
                                                         self.swing_foot_frame,
                                                         [0,0,0],
                                                         self.plant.world_frame())
        swing_target = (1 - alpha) * p_swing_current + (alpha) * primary_swing_target

        # left foot in swing
        if self.num_steps % 2 == 0:
            p_right = stance_target
            p_left = swing_target
        # right foot in swing
        else:
            p_right = swing_target
            p_left = stance_target

        self.meshcat.SetTransform("p_right", RigidTransform([p_right[0][0], -.0485, p_right[2][0]]), self.t_current)        
        self.meshcat.SetTransform("p_left", RigidTransform([p_left[0][0], 0.0485, p_left[2][0]]), self.t_current)

        return p_right, p_left
    
    # ---------------------------------------------------------------------------------------- #
    # given desired foot and torso positions, solve the IK problem
    def DoInverseKinematics(self, p_torso, p_right, p_left):

        # Update constraint on the torso position
        p_torso_lb = p_torso - self.tol_base
        p_torso_ub = p_torso + self.tol_base
        self.p_torso_cons.evaluator().UpdateLowerBound(p_torso_lb)
        self.p_torso_cons.evaluator().UpdateUpperBound(p_torso_ub)

        # Update constraints on the positions of the feet
        p_left_lb = p_left - self.tol_feet
        p_left_ub = p_left + self.tol_feet
        p_right_lb = p_right - self.tol_feet
        p_right_ub = p_right + self.tol_feet
        self.p_left_cons.evaluator().UpdateLowerBound(p_left_lb)
        self.p_left_cons.evaluator().UpdateUpperBound(p_left_ub)
        self.p_right_cons.evaluator().UpdateLowerBound(p_right_lb)
        self.p_right_cons.evaluator().UpdateUpperBound(p_right_ub)

        # solve the IK problem        
        intial_guess = self.plant.GetPositions(self.plant_context)
        self.ik.prog().SetInitialGuess(self.ik.q(), intial_guess)
        res = SnoptSolver().Solve(self.ik.prog())
        assert res.is_success(), "Inverse Kinematics Failed!"
        
        return res.GetSolution(self.ik.q())       

    # ---------------------------------------------------------------------------------------- #
    def CalcOutput(self, context, output):

        # set our interal model to match the state estimate
        x_hat = self.EvalVectorInput(context, 0).get_value()
        self.plant.SetPositionsAndVelocities(self.plant_context, x_hat)
        self.t_current = context.get_time()

        # update everything
        self.update_foot_role()
        self.update_hlip_state_R()

        # query the relevant current positions
        p_torso_current = self.plant.CalcPointsPositions(self.plant_context,
                                                         self.torso_frame,
                                                         [0,0,0],
                                                         self.plant.world_frame())

        # compute desired foot trajectories
        p_right, p_left = self.update_foot_traj()

        # solve the inverse kinematics problem
        p_torso_des = np.array([p_torso_current[0], p_torso_current[1], [self.z_nom - self.p_com_B[2]]])
        p_right_des = np.array([p_right[0], [0], p_right[2]])
        p_left_des = np.array([p_left[0], [0], p_left[2]])

        # solve the IK problem
        q_ik = self.DoInverseKinematics(p_torso_des, 
                                        p_right_des, 
                                        p_left_des)

        # compute the nominal state
        q_des = np.array([q_ik[3], q_ik[4], q_ik[5], # left leg: hip_pitch, knee, ankle
                          0 , 0,                     # left arm: shoulder pitch, elbow
                          q_ik[8], q_ik[9], q_ik[10], # right leg: hip_pitch, knee, ankle
                          0, 0])                     # right arm: shoulder pitch, elbow     
        v_des = np.zeros(6)
        x_des = np.block([q_des, v_des])

        output.SetFromVector(x_des)



#!/usr/bin/env python3

from pydrake.all import *
import numpy as np
import scipy as sp

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

        self.p_control_stance_W = np.array([0, 0, 0]).reshape(3,1)
        self.p_swing_init_W = np.array([0, 0, 0]).reshape(3,1)

        # walking parameters
        self.z_nom = 0.64  # nominal height of the CoM
        self.T_SSP = 0.3   # single support phase
        self.T_DSP = 0.0   # double support phase
        self.T = self.T_SSP + self.T_DSP

        # timing variables
        self.t_phase = 0.0

        # HLIP parameters
        g = 9.81
        self.lam = np.sqrt(g/self.z_nom)       # natural frequency
        self.A = np.array([[0,           1],   # LIP drift matrix
                           [self.lam**2, 0]])
        
        # Robot LIP state
        self.p_R = None  # center of mass position (in control stance foot frame)
        self.v_R = None  # center of mass velocity (in control stance foot frame)
        self.p_R_minus = None  # preimpact state
        self.v_R_minus = None  # preimpact state

        # ROM HLIP state
        self.p_H_minus = None
        self.v_H_minus = None

        # swing foot parameters
        self.z_apex = 0.1     # height of the apex of the swing foot 
        self.z_offset = 0.0   # offset of the swing foot from the ground
        self.z0 = 0.0
        self.zf = 0.0

        # bezier curve
        self.bezier_curve = None
        self.bez_order = 7  # 5 or 7

        # create lambda function for hyperbolic trig
        coth = lambda x: (np.exp(2 * x) + 1) / (np.exp(2 * x) - 1)

        # define the deadbeat gains and orbital slopes
        self.Kp_db = 1
        self.Kd_db = self.T_DSP + (1/self.lam) * coth(self.lam * self.T_SSP)  # deadbeat gains      
        self.sigma_P1 = self.lam * coth(0.5 * self.lam * self.T_SSP)          # orbital slope (P1)
        self.u_ff = 0.0
        self.v_des = 0.0
        self.v_max = 0.3

        # for storing the foot placement
        self.u = None

        # instantiate the inverse kinematics solver
        self.ik = InverseKinematics(self.plant)

        # inverse kinematics solver settings
        epsilon_feet = 0.00         # foot position tolerance      [m]
        epsilon_base = 0.00         # base position tolerance      [m]
        foot_epsilon_orient = 0.0   # foot orientation tolerance   [deg]
        base_epsilon_orient = 0.0   # torso orientation tolerance  [deg]
        self.tol_feet = np.array([[epsilon_feet], [np.inf], [epsilon_feet]]) 

        # Add com position constraint (fixed constraint)
        self.p_com_cons = self.ik.AddPositionConstraint(self.static_com_frame, [0, 0, 0], 
                                                        self.plant.world_frame(), 
                                                        [-np.inf, -np.inf, self.z_nom - epsilon_base], 
                                                        [np.inf, np.inf, self.z_nom + epsilon_base]) 
        
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
        self.p_H_minus = (self.v_des * self.T) / (2 + self.T_DSP * self.sigma_P1)
        self.v_H_minus = self.sigma_P1 * (self.v_des * self.T) / (2 + self.T_DSP * self.sigma_P1)

    # update the COM state of the Robot
    def update_hlip_state_R(self):

        # compute the current COM state of the robot
        p_com_W = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context).reshape(3,1)

        # copmute the current COM velocity of the robot
        J = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(self.plant_context, 
                                                                     JacobianWrtVariable.kV,
                                                                     self.plant.world_frame(), 
                                                                     self.plant.world_frame())
        v = self.plant.GetVelocities(self.plant_context)
        v_com_W = (J @ v).reshape(3,1)

        # update HLIP state of the robot
        self.p_R = p_com_W[0][0] - self.p_control_stance_W[0][0]
        self.v_R = v_com_W[0][0]
        
        # update the preipmact state of the HLIP
        x0 = np.array([self.p_R, self.v_R]).reshape(2,1)
        xt = sp.linalg.expm(self.A * (self.T_SSP - self.t_phase)) @ x0
        self.p_R_minus = xt[0][0]
        self.v_R_minus = xt[1][0]

    # -------------------------------------------------------------------------------------------------- #

    # update where to place the foot, (i.e., apply discrete control to the HLIP model)
    def update_foot_placement(self):

        # compute foot placement (local stance foot frame)
        self.u = self.u_ff + self.v_des * self.T_SSP + self.Kp_db * (self.p_R_minus - self.p_H_minus) + self.Kd_db * (self.v_R_minus - self.v_H_minus)

    # update the desired foot trajectory
    def update_foot_traj(self):

        # world frame inital and final foot placement
        u0 = self.p_swing_init_W[0][0]
        uf = self.p_control_stance_W[0][0] + self.u

        # compute the bezier curve control points
        if self.bez_order == 7:
            ctrl_pts_x = np.array([u0, u0, u0, (u0+uf)/2, uf, uf, uf])
            ctrl_pts_z = np.array([self.z0, self.z0, self.z0, (16/5)*self.z_apex, self.zf, self.zf, self.zf])
        elif self.bez_order == 5:
            ctrl_pts_x = np.array([u0, u0, (u0+uf)/2, uf, uf])
            ctrl_pts_z = np.array([self.z0, self.z0, (8/3)*self.z_apex, self.zf, self.zf])

        # build control point matrix
        ctrl_pts = np.vstack((ctrl_pts_x, 
                              ctrl_pts_z)) 

        # build the swing foot trajectory
        self.bezier_swing_traj = BezierCurve(0, self.T_SSP, ctrl_pts)

    # -------------------------------------------------------------------------------------------------- #

    # solve the inverse kinematics problem
    def solve_ik(self, p_stance, p_swing):

        # udpate the foot placement constraints depending on the stance foot
        if self.stance_foot_frame == self.left_foot_frame:
            self.p_left_cons.evaluator().UpdateLowerBound(p_stance - self.tol_feet)
            self.p_left_cons.evaluator().UpdateUpperBound(p_stance + self.tol_feet)
            self.p_right_cons.evaluator().UpdateLowerBound(p_swing - self.tol_feet)
            self.p_right_cons.evaluator().UpdateUpperBound(p_swing + self.tol_feet)
        elif self.stance_foot_frame == self.right_foot_frame:
            self.p_right_cons.evaluator().UpdateLowerBound(p_stance - self.tol_feet)
            self.p_right_cons.evaluator().UpdateUpperBound(p_stance + self.tol_feet)
            self.p_left_cons.evaluator().UpdateLowerBound(p_swing - self.tol_feet)
            self.p_left_cons.evaluator().UpdateUpperBound(p_swing + self.tol_feet)

        # solve the IK problem
        initial_guess = self.plant.GetPositions(self.plant_context)
        self.ik.prog().SetInitialGuess(self.ik.q(), initial_guess)
        res = SnoptSolver().Solve(self.ik.prog())

        return res

    # -------------------------------------------------------------------------------------------------- #

    # set the trajectpry generation problem parameters
    def set_problem_params(self, q0, v0, stance_foot, p_swing_init_W, v_des):

        # set the robot state
        self.plant.SetPositions(self.plant_context, q0)
        self.plant.SetVelocities(self.plant_context, v0)

        # set the desired velocity
        self.v_des = v_des

        # set which foot is in stance
        if stance_foot == "left_foot":
            self.stance_foot_frame = self.left_foot_frame
            p_stance_W = self.plant.CalcPointsPositions(self.plant_context, 
                                                        self.left_foot_frame, [0,0,0], 
                                                        self.plant.world_frame())
            self.p_control_stance_W = np.array([p_stance_W[0][0],  0.1, self.z_offset]).reshape(3,1)
        elif stance_foot == "right_foot":
            self.stance_foot_frame = self.right_foot_frame
            p_stance_W = self.plant.CalcPointsPositions(self.plant_context, 
                                                        self.right_foot_frame, [0,0,0], 
                                                        self.plant.world_frame())
            self.p_control_stance_W = np.array([p_stance_W[0][0], -0.1, self.z_offset]).reshape(3,1)
            
        # set the initial swing foot position
        self.p_swing_init_W = p_swing_init_W

    # main function that updates the whole problem
    def get_trajectory(self, q0, v0, stance_foot, p_swing_init_W, v_des = 0.0):
        
        # setup the problem parameters
        self.set_problem_params(q0, v0, stance_foot, p_swing_init_W, v_des)

        # update everything
        self.update_hlip_state_H()
        self.update_hlip_state_R()
        self.update_foot_placement()
        self.update_foot_traj()

        # plot the bezier curve
        times = np.linspace(0, self.T_SSP, 100)
        swing_traj = []
        for t in times:
            swing_traj.append(self.bezier_swing_traj.value(t))
        swing_traj = np.array(swing_traj)[:,:,0]

######################################################################################################################

if __name__ == "__main__":

    # model path
    model_file = "../../models/achilles_SE2_drake.urdf"

    # start meshcat
    meshcat = StartMeshcat()

    # create the trajectory generator
    g = HLIPTrajectoryGeneratorSE2(model_file)
    
    # set desired problem parameters
    v_des = 0.1
    stance_foot = "right_foot"
    q0 = np.array([0, 0.97,   # position (x,z)
                   0.2,        # theta
                   -0.96, 1.22, 0.17,  # left leg: hip_pitch, knee, ankle 
                   -0.36, 0.7, -0.54]) # right leg: hip_pitch, knee, ankle
    v0 = np.zeros(len(q0))
    v0[0] = 0.0              # forward x-velocity
    p_swing_init_W = np.array([0, 0, 0]).reshape(3,1)

    g.get_trajectory(q0 = q0, 
                     v0 = v0,
                     stance_foot = stance_foot,
                     p_swing_init_W = p_swing_init_W,
                     v_des = v_des)

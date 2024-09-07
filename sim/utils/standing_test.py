
#!/usr/bin/env python3

import numpy as np
import scipy as sp
import time
import yaml

from pydrake.all import *

import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../bin'))
import AchillesKinematicsPy as ak  # type: ignore

# import the yaml config
config_path = "../config/config.yaml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)   

#################################################################################################

class StandTest(LeafSystem):

    def __init__(self, model_file, config, q0, meshcat):
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
        self.torso_frame = self.plant.GetFrameByName("torso")

        self.left_foot_heel_frame = self.plant.GetFrameByName("left_foot_heel")
        self.left_foot_toe_frame = self.plant.GetFrameByName("left_foot_toe")
        self.right_foot_heel_frame = self.plant.GetFrameByName("right_foot_heel")
        self.right_foot_toe_frame = self.plant.GetFrameByName("right_foot_toe")

        # input port
        self.state_input_port = self.DeclareVectorInputPort("state", 
                                                            BasicVector(self.plant.num_positions() + self.plant.num_velocities()))
        self.DeclareVectorOutputPort("x_des",
                                     BasicVector(2 * self.plant.num_actuators()),
                                     self.CalcOutput)        
        self.q0 = q0

        # compute fixed distances
        q_zeros = np.zeros(self.plant.num_positions())
        q_zeros[0]= 1
        self.plant.SetPositions(self.plant_context, q_zeros)

        foot_length = self.plant.CalcPointsPositions(self.plant_context,
                                                          self.right_foot_toe_frame, np.zeros(3),
                                                          self.right_foot_heel_frame)
        self.foot_len = foot_length[0][0] # supposed to be 117.91 [mm]
    
        foot_to_foot_distance = self.plant.CalcPointsPositions(self.plant_context,
                                                               self.left_foot_heel_frame, np.zeros(3),
                                                               self.right_foot_heel_frame)
        self.foot_to_foot_distance = foot_to_foot_distance[1][0] # supposed to be 180 [mm]

        self.heel_to_foot_frame_distance_z = 0.01259 # [m]
        self.heel_to_foot_frame_distance_x = 0.01859 # [m]

        # instantiate inverse kinematics solver
        self.ik = InverseKinematics(self.plant)

        # inverse kinematics solver settings
        self.epsilon_feet = 0.00     # foot position tolerance     [m]
        self.epsilon_base = 0.00     # torso position tolerance    [m]
        self.foot_epsilon_orient = 0.0   # foot orientation tolerance  [deg]
        self.base_epsilon_orient = 0.0   # torso orientation tolerance [deg]
        self.tol_base = np.array([[self.epsilon_base], [self.epsilon_base], [self.epsilon_base]])
        self.tol_feet = np.array([[self.epsilon_feet], [self.epsilon_feet], [self.epsilon_feet]])

        # Add com position constraint (fixed constraint)
        self.p_left_cons =  self.ik.AddPositionConstraint(self.left_foot_frame, [0, 0, 0],
                                                          self.static_com_frame, 
                                                          [0, 0, 0], [0, 0, 0])
        self.p_right_cons = self.ik.AddPositionConstraint(self.right_foot_frame, [0, 0, 0],
                                                          self.static_com_frame, 
                                                          [0, 0, 0], [0, 0, 0]) 
        
        # Add foot orientation constraints, aligns toe directions (remove and create this constraint at every discrete S2S step)
        self.r_left_cons = self.ik.AddAngleBetweenVectorsConstraint(self.left_foot_frame, [1, 0, 0],
                                                                    self.static_com_frame, [1, 0, 0],
                                                                    0, 0)
        self.r_right_cons = self.ik.AddAngleBetweenVectorsConstraint(self.right_foot_frame, [1, 0, 0],
                                                                    self.static_com_frame, [1, 0, 0],
                                                                    0, 0)
        
        self.meshcat = meshcat
        self.meshcat.SetObject("com", Sphere(0.01), Rgba(0, 1, 0, 1))
        
    def DoCalculations(self):
        
        p_left_com = np.array([-0.01, 0.09, -0.64]).reshape(3,1)
        p_right_com = np.array([-0.01, -0.09, -0.64]).reshape(3,1)

        # update the com position constraints
        p_left_lb = p_left_com - self.tol_feet
        p_left_ub = p_left_com + self.tol_feet
        self.p_left_cons.evaluator().set_bounds(p_left_lb, p_left_ub)

        p_right_lb = p_right_com - self.tol_feet
        p_right_ub = p_right_com + self.tol_feet
        self.p_right_cons.evaluator().set_bounds(p_right_lb, p_right_ub)

        self.ik.prog().SetInitialGuess(self.ik.q(), self.q0)
        res = SnoptSolver().Solve(self.ik.prog())

        return res

    def com_computaion(self, t):

        p_com_frame = self.plant.CalcPointsPositions(self.plant_context,
                                                     self.static_com_frame, np.zeros(3),
                                                     self.plant.world_frame())
        p_com_torso = self.plant.CalcPointsPositions(self.plant_context,
                                                     self.torso_frame, np.zeros(3),
                                                     self.static_com_frame)

        # compute the center of mass in world frame
        com = self.plant.CalcCenterOfMassPositionInWorld(self.plant_context).reshape(3,1)
        com_vis = np.array([com[0], com[1], [0]])
        com_error = p_com_frame - com
        self.meshcat.SetTransform("com", RigidTransform(com_vis), t)

        p_com_actual_torso = self.plant.CalcPointsPositions(self.plant_context,
                                                            self.plant.world_frame(), com,
                                                            self.torso_frame)
        # print("p_com_actual_torso: ", p_com_actual_torso.T)
        # print("com error: ", com_error)

    def CalcOutput(self, context, output):
        
        # Get the current state
        x0 = self.state_input_port.Eval(context)
        self.plant.SetPositionsAndVelocities(self.plant_context, x0)
        q0 = x0[:self.plant.num_positions()]

        print(", ".join([f"{x:.4f}" for x in q0]))

        self.com_computaion(context.get_time())
        res = self.DoCalculations()
        q_ik = res.GetSolution(self.ik.q())

        # insert the arms
        q_des = q_ik[7:]
        # print("q_des: ", q_des)
        v_des = np.zeros(self.plant.num_actuators())
        x_des = np.block([q_des, v_des])

        output.SetFromVector(x_des)

#################################################################################################

# model file
if __name__ == "__main__":

    q0 = np.array(
[ 1.00000879e+00,  1.81677219e-12, -3.06192039e-03,  1.95087574e-10,
 -6.26592809e-03,  6.07159757e-11,  9.50145419e-01,  2.10129417e-06,
 -2.13802235e-04, -4.75199896e-01,  1.00120324e+00, -5.24557240e-01,
  5.07670474e-05, -6.91523929e-19, -6.19850681e-19,  4.57014363e-09,
 -2.09707577e-06,  2.13802332e-04, -4.75199896e-01,  1.00120324e+00,
 -5.24557240e-01,  5.07670474e-05, -1.05951601e-14, -6.20021438e-19,
  4.57014363e-09]

)            # right arm, (shoulder pitch, shoulder roll, shoulder yaw, elbow)

    # start meshcat
    meshcat = StartMeshcat()

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

    # Add implicit PD controllers (must use kLagged or kSimilar)
    Kp = np.array(config['gains']['Kp']) * 3.0
    Kd = np.array(config['gains']['Kd']) * 1.0 
    actuator_indices = [JointActuatorIndex(i) for i in range(plant.num_actuators())]
    for actuator_index, Kp, Kd in zip(actuator_indices, Kp, Kd):
        plant.get_joint_actuator(actuator_index).set_controller_gains(
            PdControllerGains(p=Kp, d=Kd))    
        
    plant.gravity_field().set_gravity_vector([0, 0, -9.81])

    plant.Finalize()

    # controller
    controller = builder.AddSystem(StandTest(model_file, config, q0, meshcat))
    builder.Connect(plant.get_state_output_port(), 
            controller.GetInputPort("state"))
    builder.Connect(controller.GetOutputPort("x_des"),
            plant.get_desired_state_input_port(models[0]))
    
    AddDefaultVisualization(builder,meshcat)

    # Build the system diagram
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

    # set the initial state
    v0 = np.array(config['v0'])
    plant.SetPositions(plant_context, q0)
    plant.SetVelocities(plant_context, v0)

    simulator = Simulator(diagram, diagram_context)
    simulator.set_target_realtime_rate(1.0)
    simulator.Initialize()

    # Simulate and play back on meshcat
    meshcat.StartRecording()
    simulator.AdvanceTo(15.0)
    meshcat.StopRecording()
    meshcat.PublishRecording()
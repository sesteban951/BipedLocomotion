
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

    def __init__(self, model_file, config, q0):
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

        print("Fixed distances:")
        print(self.foot_len, self.foot_to_foot_distance)

    def DoCalculations(self):

        p_left_heel = self.plant.CalcPointsPositions(self.plant_context, 
                                                     self.left_foot_heel_frame, np.zeros(3),
                                                     self.plant.world_frame())
        p_left_toe = self.plant.CalcPointsPositions(self.plant_context,
                                                    self.left_foot_toe_frame, np.zeros(3),
                                                    self.plant.world_frame())

    def CalcOutput(self, context, output):
        
        # Get the current state
        x0 = self.state_input_port.Eval(context)
        self.plant.SetPositionsAndVelocities(self.plant_context, x0)

        self.DoCalculations()

        # insert the arms
        q_des = q0[7:]
        v_des = np.zeros(self.plant.num_actuators())
        x_des = np.block([q_des, v_des])

        output.SetFromVector(x_des)

#################################################################################################

# model file
if __name__ == "__main__":

    q0 = np.array([1.0000, 0.0000, 0.0000, 0.0000,            # base orientation, (w, x, y, z)
                   0.0000, 0.0000, 0.9500,                    # base position, (x,y,z)
                   0.0000, 0.0209, -0.5515, 1.0239,-0.4725,   # left leg, (hip yaw, hip roll, hip pitch, knee, ankle) 
                   0.0900, 0.000, 0.0000, -0.0000,            # left arm, (shoulder pitch, shoulder roll, shoulder yaw, elbow)
                   0.0000, -0.0209, -0.5515, 1.0239,-0.4725,  # right leg, (hip yaw, hip roll, hip pitch, knee, ankle) 
                   0.0900, 0.000, 0.0000, -0.0000])            # right arm, (shoulder pitch, shoulder roll, shoulder yaw, elbow)

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
    Kp = np.array(config['gains']['Kp'])
    Kd = np.array(config['gains']['Kd'])    
    actuator_indices = [JointActuatorIndex(i) for i in range(plant.num_actuators())]
    for actuator_index, Kp, Kd in zip(actuator_indices, Kp, Kd):
        plant.get_joint_actuator(actuator_index).set_controller_gains(
            PdControllerGains(p=Kp, d=Kd))    
        
    plant.gravity_field().set_gravity_vector([0, 0, -0.81])

    plant.Finalize()

    # controller
    controller = builder.AddSystem(StandTest(model_file, config, q0))
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
    simulator.set_target_realtime_rate(0.5)
    simulator.Initialize()

    # Simulate and play back on meshcat
    meshcat.StartRecording()
    simulator.AdvanceTo(15.0)
    meshcat.StopRecording()
    meshcat.PublishRecording()

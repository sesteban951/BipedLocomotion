#!/usr/bin/env python3

from pydrake.all import *
import numpy as np
from controller_SE2 import HLIP

# simulation parameters
sim_time = 30.
realtime_rate = 1.0

# load model
# model_file = "../models/achilles_drake.urdf"
model_file = "../models/achilles_SE2_drake.urdf"

# start meshcat
meshcat = StartMeshcat()

# simulation parameters
sim_hz = 1000
sim_config = MultibodyPlantConfig()
sim_config.time_step = 1 / sim_hz 
sim_config.discrete_contact_approximation = "lagged"
sim_config.contact_model = "hydroelastic_with_fallback"

# Set up the Drake system diagram
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlant(sim_config, builder)

# Add the harpy model
robot = Parser(plant).AddModels(model_file)[0]

# Add the ground
plant.RegisterCollisionGeometry(
    plant.world_body(),
    RigidTransform(p = [0,0,-25]),
    Box(50,50,50), "ground",
    CoulombFriction(0.7, 0.7))

# add gravity
plant.gravity_field().set_gravity_vector([0, 0, -9.81])

# add low level PD controllers
kp_hip = 500
kp_knee = 500
kp_ankle = 200
kd_hip = 10
kd_knee = 10
kd_ankle = 2
Kp = np.array([kp_hip, kp_knee, kp_ankle, kp_hip, kp_knee, kp_ankle])
Kd = np.array([kd_hip, kd_knee, kd_ankle, kd_hip, kd_knee, kd_ankle])
actuator_indices = [JointActuatorIndex(i) for i in range(plant.num_actuators())]
for actuator_index, Kp, Kd in zip(actuator_indices, Kp, Kd):
    plant.get_joint_actuator(actuator_index).set_controller_gains(
    PdControllerGains(p=Kp, d=Kd)
)

# finalize the plant
plant.Finalize()

# add the controller
controller = builder.AddSystem(HLIP(model_file, meshcat))

# build the diagram 
builder.Connect(plant.get_state_output_port(), 
                controller.GetInputPort("x_hat"))
builder.Connect(controller.GetOutputPort("x_des"),
                plant.get_desired_state_input_port(robot))

# add the visualizer
AddDefaultVisualization(builder, meshcat)

# build the complete diagram
diagram = builder.Build()

# Create a context for this diagram and plant
diagram_context = diagram.CreateDefaultContext()
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)

# configuration 
# q0 = np.array([1, 0, 0, 0,    # quaternion (w,x,y,z)
#                0, 0, 1.02,    # position (x,y,z)
#                0, 0, 0, 0, 0, # left leg: hip_yaw, hip_roll, hip_pitch, knee, ankle 
#                0, 0, 0, 0,    # left arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
#                0, 0, 0, 0, 0, # right leg: hip_yaw, hip_roll, hip_pitch, knee, ankle
#                0, 0, 0, 0])   # right arm: shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
# q0 = np.array([0, 1.02,  # position (x,z)
#                0,        # theta
#                0, 0, 0,  # left leg: hip_pitch, knee, ankle 
#                0, 0,     # left arm: shoulder_pitch, elbow
#                0, 0, 0,  # right leg: hip_pitch, knee, ankle
#                0, 0])    # right arm: shoulder_pitch, elbow
q0 = np.array([0, 1.,  # position (x,z)
               0,        # theta
               0, 0, 0,  # left leg: hip_pitch, knee, ankle 
               0, 0, 0]) # right leg: hip_pitch, knee, ankle
v0 = np.zeros(plant.num_velocities())
plant.SetPositions(plant_context, q0)
plant.SetVelocities(plant_context, v0)

# Run the Drake diagram with a Simulator object
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(realtime_rate)
simulator.Initialize()

# run the sim and record
meshcat.StartRecording()
simulator.AdvanceTo(sim_time)
meshcat.StopRecording()
meshcat.PublishRecording()
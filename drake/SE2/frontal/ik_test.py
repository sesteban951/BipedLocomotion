#!/usr/bin/env python3

from pydrake.all import *
import numpy as np
import time

# load model
model_file = "../../../models/achilles_SE2_drake_frontal.urdf"

# create internal model of the robot
plant = MultibodyPlant(0)
Parser(plant).AddModels(model_file)
plant.Finalize()
plant_context = plant.CreateDefaultContext()

# relevant frames
static_com_frame = plant.GetFrameByName("static_com") # nominal is 0.734 z in world frame
left_foot_frame = plant.GetFrameByName("left_foot")
right_foot_frame = plant.GetFrameByName("right_foot")

# instantiate inverse kinematics solver
ik = InverseKinematics(plant)
q_ik_sol = np.zeros(plant.num_positions())

# inverse kinematics solver settings
epsilon_feet = 0.005     # foot position tolerance     [m]
epsilon_base = 0.005     # torso position tolerance    [m]
foot_epsilon_orient = 1.0   # foot orientation tolerance  [deg]
base_epsilon_orient = 1.0   # torso orientation tolerance [deg]
tol_base = np.array([[np.inf], [epsilon_base], [epsilon_base]])  # y-z only
tol_feet = np.array([[np.inf], [np.inf], [epsilon_feet]])  # y-z only

# Add com position constraint
p_com_cons = ik.AddPositionConstraint(static_com_frame, [0, 0, 0], 
                                      plant.world_frame(), 
                                      [0, 0, 0], [0, 0, 0]) 

# Add com orientation constraint
r_com_cons = ik.AddOrientationConstraint(static_com_frame, RotationMatrix(),
                                         plant.world_frame(), RotationMatrix(),
                                         base_epsilon_orient * (np.pi/180))

# Add foot position constraints
p_left_cons =  ik.AddPositionConstraint(left_foot_frame, [0, 0, 0],
                                        plant.world_frame(), 
                                        [0, 0, 0], [0, 0, 0])
p_right_cons = ik.AddPositionConstraint(right_foot_frame, [0, 0, 0],
                                        plant.world_frame(), 
                                        [0, 0, 0], [0, 0, 0]) 

# Add foot orientation constraints
r_left_cons =  ik.AddOrientationConstraint(left_foot_frame, RotationMatrix(),
                                           plant.world_frame(), RotationMatrix(),
                                           foot_epsilon_orient * (np.pi/180))
r_right_cons = ik.AddOrientationConstraint(right_foot_frame, RotationMatrix(),
                                           plant.world_frame(), RotationMatrix(),
                                           foot_epsilon_orient * (np.pi/180))

# solve the IK problem 
p_static_com_W = plant.CalcPointsPositions(plant_context, 
                                           static_com_frame, 
                                           [0, 0, 0], 
                                           plant.world_frame())
p_left_target = plant.CalcPointsPositions(plant_context, 
                                          left_foot_frame, 
                                          [0, 0, 0], 
                                          plant.world_frame())
p_right_target = plant.CalcPointsPositions(plant_context, 
                                           right_foot_frame, 
                                           [0, 0, 0], 
                                           plant.world_frame())
p_com_target = p_static_com_W
p_com_target  = np.array([0.0, 0., 0.64]).reshape(3,1)
p_right_target = np.array([-0, -0.1, 0.]).reshape(3,1)
p_left_target = np.array([0, 0.1, 0.]).reshape(3,1)

p_com_cons.evaluator().UpdateLowerBound(p_com_target - tol_base)
p_com_cons.evaluator().UpdateUpperBound(p_com_target + tol_base)
p_left_cons.evaluator().UpdateLowerBound(p_left_target - tol_feet)
p_left_cons.evaluator().UpdateUpperBound(p_left_target + tol_feet)
p_right_cons.evaluator().UpdateLowerBound(p_right_target - tol_feet)
p_right_cons.evaluator().UpdateUpperBound(p_right_target + tol_feet)

# solve the IK problem
res = SnoptSolver().Solve(ik.prog())

if res.is_success():

    q_ik_sol = res.GetSolution(ik.q())
    print("Good.")

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

    # plot the desired target positions
    meshcat.SetObject("target/com", Sphere(0.05), Rgba(0, 1, 0,1))
    meshcat.SetObject("target/left", Sphere(0.05), Rgba(0, 0, 1,1))
    meshcat.SetObject("target/right", Sphere(0.05), Rgba(1, 0, 0,1))

    meshcat.SetTransform("target/com", RigidTransform(p_com_target))
    meshcat.SetTransform("target/left", RigidTransform(p_left_target))
    meshcat.SetTransform("target/right", RigidTransform(p_right_target))

    # Set the Drake model to have this state
    q0 = q_ik_sol
    plant.SetPositions(plant_context, q0)

    # Perform a forced publish event. This will propagate the plant's state to 
    # meshcat, without doing any physics simulation.
    diagram.ForcedPublish(diagram_context)

    time.sleep(5)

    # Publish the meshcat recording
    meshcat.StopRecording()
    meshcat.PublishRecording()


else: 
    print("Bad.")














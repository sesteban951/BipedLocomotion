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

# set teh intial condition
q0 = np.array([0, 1.0, 
               0, 
               0, 0, 0, 0, 
               0, 0, 0, 0])  # initial condition

# set plant configruation 
plant.SetPositions(plant_context, q0)

# relevant frames
static_com_frame = plant.GetFrameByName("static_com") # nominal is 0.734 z in world frame
left_foot_frame = plant.GetFrameByName("left_foot")
right_foot_frame = plant.GetFrameByName("right_foot")

#############################################################################################

# instantiate inverse kinematics solver
ik = InverseKinematics(plant, with_joint_limits=True)

# add a quadratic error cost to the solver
# cost = 0.5 * (q_hat - q_nominal)^T * Q * (q_hat - q_nominal)
# no arms (parallel legs, bent knees)
q_nominal = np.array([
    0.0000, 0.9300,           # base position
    0.0000,                   # base orientation
    0, -0.5515, 1.0239,-0.4725,  # left leg
    0, -0.5515, 1.0239,-0.4725,  # left leg
])
# Q = np.diag([0, 0,
#              1, 
#              1, 1, 1, 1, 
#              1, 1, 1, 1]) * 1
# ik.prog().AddQuadraticErrorCost(Q=np.eye(plant.num_positions()), 
#                                 x_desired=q_nominal, vars=ik.q())

# # inverse kinematics solver settings
# epsilon_feet = 0.003     # foot position tolerance     [m]
# epsilon_base = 0.003     # torso position tolerance    [m]
# foot_epsilon_orient = 1.0   # foot orientation tolerance  [deg]
# base_epsilon_orient = 1.0   # torso orientation tolerance [deg]
# tol_base = np.array([[np.inf], [epsilon_base], [epsilon_base]])  # y-z only
# tol_feet = np.array([[epsilon_feet], [epsilon_feet], [epsilon_feet]])  # y-z only

# t1 = time.time()
# # Add com position constraint
# p_com_cons = ik.AddPositionConstraint(static_com_frame, [0, 0, 0], 
#                                       plant.world_frame(), 
#                                       [0,0,0], [0,0,0])

# # Add com orientation constraint
# rpy_base = RollPitchYaw([0, 0, 0])
# R_base = RotationMatrix(rpy_base)
# r_com_cons = ik.AddOrientationConstraint(static_com_frame, RotationMatrix(),
#                                          plant.world_frame(), R_base,
#                                          base_epsilon_orient * (np.pi/180))

# # Add foot position constraints
# p_left_cons = ik.AddPositionConstraint(left_foot_frame, [0, 0, 0],
#                                         plant.world_frame(), 
#                                         [0,0,0], [0,0,0])
# p_right_cons = ik.AddPositionConstraint(right_foot_frame, [0, 0, 0],
#                                         plant.world_frame(), 
#                                         [0,0,0], [0,0,0])

# # Add foot orientation constraints
# r_right_cons = ik.AddAngleBetweenVectorsConstraint(right_foot_frame, [0, 0, 1],
#                                                    plant.world_frame(), [0, 0, 1],
#                                                    0, foot_epsilon_orient * (np.pi/180))
# r_left_cons = ik.AddAngleBetweenVectorsConstraint(left_foot_frame, [0, 0, 1],
#                                                   plant.world_frame(), [0, 0, 1],
#                                                   0, foot_epsilon_orient * (np.pi/180))
# t2 = time.time()
# T1 = t2-t1

#############################################################################################

# set the desired target positions
p_com_target  = np.array([0.0, .0, 0.75]).reshape(3,1)
p_left_target = np.array([0, 0.1, 0.20]).reshape(3,1)
p_right_target = np.array([0, -0.1, 0.20]).reshape(3,1)

# # update the constraints
# t1 = time.time()
# p_com_cons.evaluator().UpdateLowerBound(p_com_target - tol_base)
# p_com_cons.evaluator().UpdateUpperBound(p_com_target + tol_base)
# p_left_cons.evaluator().UpdateLowerBound(p_left_target - tol_feet)
# p_left_cons.evaluator().UpdateUpperBound(p_left_target + tol_feet)
# p_right_cons.evaluator().UpdateLowerBound(p_right_target - tol_feet)
# p_right_cons.evaluator().UpdateUpperBound(p_right_target + tol_feet)
# t2 = time.time()
# T2 = t2-t1
# print("Time to create and update hard constraints: ", T1, T2)

# add the costs
t0 = time.time()
C = np.diag([1, 1, 1])*10
p_com_cons = ik.AddPositionCost(static_com_frame, [0, 0, 0],
                                plant.world_frame(), p_com_target,
                                C)

C = 10.0
r_com_cons = ik.AddOrientationCost(static_com_frame, RotationMatrix(),
                                   plant.world_frame(), RotationMatrix(),
                                   C)

C = np.diag([1, 1, 1])*10
p_left_cons = ik.AddPositionCost(left_foot_frame, [0, 0, 0],
                                plant.world_frame(), p_left_target,
                                C)
p_right_cons = ik.AddPositionCost(right_foot_frame, [0, 0, 0],
                                plant.world_frame(), p_right_target,
                                C)

C = 10.0
r_right_cons = ik.AddAngleBetweenVectorsCost(right_foot_frame, [0, 0, 1],
                                            plant.world_frame(), [0, 0, 1],
                                            C)
r_left_cons = ik.AddAngleBetweenVectorsCost(left_foot_frame, [0, 0, 1],
                                            plant.world_frame(), [0, 0, 1],
                                            C)
t1 = time.time()
print("Add costs time: ", t1-t0)

# solve the IK problem
# refer to https://web.stanford.edu/group/SOL/guides/sndoc7.pdf for solver options
solver = SnoptSolver()
solver_options = SolverOptions()
solver_options.SetOption(solver.solver_id(), "Major optimality tolerance", 1e-4)  # Example tolerance
solver_options.SetOption(solver.solver_id(), "Linesearch tolerance", 0.9)

t0 = time.time()
ik.prog().SetInitialGuess(ik.q(), q_nominal)
res = solver.Solve(ik.prog(), None, solver_options)
# res = SnoptSolver().Solve(ik.prog())
t1 = time.time()
print("IK solve time: ", t1-t0)

if res.is_success():

    q_ik_sol = res.GetSolution(ik.q())
    print("Good.")
    print("q_ik_sol: ", q_ik_sol)

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

    q0 = q_ik_sol
    plant.SetPositions(plant_context, q0)

    p_right = plant.CalcPointsPositions(plant_context, 
                                           right_foot_frame, 
                                           [0, 0, 0], 
                                           plant.world_frame())

    # Start meshcat recording
    meshcat.StartRecording()

    # plot the desired target positions
    meshcat.SetObject("target/com", Sphere(0.025), Rgba(0, 1, 0,1))
    # meshcat.SetObject("target/left", Sphere(0.025), Rgba(0, 0, 1,1))
    meshcat.SetObject("target/right", Sphere(0.025), Rgba(1, 0, 0,1))

    meshcat.SetTransform("target/com", RigidTransform(p_com_target))
    # meshcat.SetTransform("target/left", RigidTransform(p_left_target))
    meshcat.SetTransform("target/right", RigidTransform(p_right_target))

    # Set the Drake model to have this stat
    # print("p_right_current: ", p_right)

    # Perform a forced publish event. This will propagate the plant's state to 
    # meshcat, without doing any physics simulation.
    diagram.ForcedPublish(diagram_context)

    time.sleep(5)

    # Publish the meshcat recording
    meshcat.StopRecording()
    meshcat.PublishRecording()


else: 
    print("Bad.")
    a = res.GetInfeasibleConstraints(ik.prog())
    n = res.GetInfeasibleConstraintNames(ik.prog())
    print("\n ************* IK failed! ************* \n")
    # print("Infeasible constraints: ", a)
    print("Infeasible constraint names: ", n)

    # start meshcat
    meshcat = StartMeshcat()

    # Start meshcat recording
    meshcat.StartRecording()

    # plot the desired target positions
    meshcat.SetObject("target/com", Sphere(0.025), Rgba(0, 1, 0,1))
    # meshcat.SetObject("target/left", Sphere(0.025), Rgba(0, 0, 1,1))
    meshcat.SetObject("target/right", Sphere(0.025), Rgba(1, 0, 0,1))

    meshcat.SetTransform("target/com", RigidTransform(p_com_target))
    # meshcat.SetTransform("target/left", RigidTransform(p_left_target))
    meshcat.SetTransform("target/right", RigidTransform(p_right_target))

    time.sleep(5)

    # Publish the meshcat recording
    meshcat.StopRecording()
    meshcat.PublishRecording()


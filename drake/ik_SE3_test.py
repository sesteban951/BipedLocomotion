from pydrake.all import *
import numpy as np

# ik testing controller
class Controller(LeafSystem):

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
        self.magenta_color = Rgba(1, 0, 1, 1)
        self.sphere = Sphere(0.025)

        # self.meshcat.SetObject("p_right", self.sphere, self.red_color)
        # self.meshcat.SetObject("p_left", self.sphere, self.blue_color)
        self.meshcat.SetObject("p_stance", self.sphere, self.green_color)
        self.meshcat.SetObject("p_swing", self.sphere, self.magenta_color)

        # create internal model of the robot
        self.plant = MultibodyPlant(0)
        Parser(self.plant).AddModels(model_file)
    
        # relevant frames
        self.left_foot_frame = self.plant.GetFrameByName("left_foot")
        self.right_foot_frame = self.plant.GetFrameByName("right_foot")
        self.stance_foot_frame = self.right_foot_frame
        self.swing_foot_frame  = self.left_foot_frame
        self.stance_foot_control_frame = FixedOffsetFrame("stance_foot_control_frame", 
                                                          self.stance_foot_frame, 
                                                          RigidTransform())
        self.swing_foot_control_frame = FixedOffsetFrame("swing_foot_control_frame",
                                                         self.swing_foot_frame,
                                                         RigidTransform())
        self.plant.AddFrame(self.stance_foot_control_frame)
        self.plant.AddFrame(self.swing_foot_control_frame)

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

        # instantiate inverse kinematics solver
        self.ik = InverseKinematics(self.plant)

        # inverse kinematics solver settings
        epsilon_feet = 0.00        # foot position tolerance     [m]
        foot_epsilon_orient = 0.   # foot orientation tolerance  [deg]
        self.tol_feet = np.array([[epsilon_feet], [epsilon_feet], [epsilon_feet]])  # x-z only
        
        # Add foot position constraints
        # self.p_left_cons =  self.ik.AddPositionConstraint(self.left_foot_frame, [0, 0, 0],
        #                                                   self.plant.world_frame(), 
        #                                                   [0, 0, 0], [0, 0, 0])
        # self.p_right_cons = self.ik.AddPositionConstraint(self.right_foot_frame, [0, 0, 0],
        #                                                   self.plant.world_frame(), 
        #                                                   [0, 0, 0], [0, 0, 0]) 
        self.p_swing_foot_cons = self.ik.AddPositionConstraint(self.swing_foot_frame, [0, 0, 0],
                                                               self.stance_foot_control_frame,
                                                               [0, 0, 0], [0, 0, 0])
        self.p_stance_foot_cons = self.ik.AddPositionConstraint(self.stance_foot_frame, [0, 0, 0],
                                                               self.plant.world_frame(),
                                                               [0, 0, 0], [0, 0, 0])
        
        # Add foot orientation constraints
        # self.r_left_cons =  self.ik.AddOrientationConstraint(self.left_foot_frame, RotationMatrix(),
        #                                                      self.plant.world_frame(), RotationMatrix(),
        #                                                      foot_epsilon_orient * (np.pi/180))
        # self.r_right_cons = self.ik.AddOrientationConstraint(self.right_foot_frame, RotationMatrix(),
        #                                                      self.plant.world_frame(), RotationMatrix(),
        #                                                      foot_epsilon_orient * (np.pi/180))
        self.r_swing_foot_cons = self.ik.AddOrientationConstraint(self.swing_foot_frame, RotationMatrix(),
                                                                  self.stance_foot_control_frame, RotationMatrix(),
                                                                  foot_epsilon_orient * (np.pi/180))
        Rot_W = RotationMatrix(RollPitchYaw(0, 1, 0))
        self.r_stance_foot_cons = self.ik.AddOrientationConstraint(self.stance_foot_frame, RotationMatrix(),
                                                                   self.plant.world_frame(), Rot_W,
                                                                   foot_epsilon_orient * (np.pi/180))

    # ---------------------------------------------------------------------------------------- #
    # given desired foot and torso positions, solve the IK problem
    # def DoInverseKinematics(self, p_right, p_left):

    #     # Update constraints on the positions of the feet
    #     p_left_lb = p_left - self.tol_feet
    #     p_left_ub = p_left + self.tol_feet
    #     p_right_lb = p_right - self.tol_feet
    #     p_right_ub = p_right + self.tol_feet
    #     self.p_left_cons.evaluator().UpdateLowerBound(p_left_lb)
    #     self.p_left_cons.evaluator().UpdateUpperBound(p_left_ub)
    #     self.p_right_cons.evaluator().UpdateLowerBound(p_right_lb)
    #     self.p_right_cons.evaluator().UpdateUpperBound(p_right_ub)

    #     # solve the IK problem        
    #     intial_guess = self.plant.GetPositions(self.plant_context)
    #     self.ik.prog().SetInitialGuess(self.ik.q(), intial_guess)
    #     res = SnoptSolver().Solve(self.ik.prog())
        
    #     return res
    
    def DoInverseKinematics(self, p_stance, p_swing):

        # update the stance foot control frame orientation 
        R_W = self.plant.CalcRelativeRotationMatrix(self.plant_context,
                                                    self.plant.world_frame(),
                                                    self.stance_foot_frame)
        rpy = R_W.ToRollPitchYaw()
        pitch = rpy.pitch_angle()
        print(pitch)
        rpy = RollPitchYaw(0, -pitch, 0)
        R_new = RotationMatrix(rpy)
        self.stance_foot_control_frame.SetPoseInParentFrame(self.plant_context, 
                                                            RigidTransform(R_new, [0, 0, 0]))
        R = self.plant.CalcRelativeRotationMatrix(self.plant_context,
                                                    self.plant.world_frame(),
                                                    self.stance_foot_control_frame) # <----- here is where you adjust the control frame pitch to be aligned with the world
        print(R.ToRollPitchYaw().pitch_angle())

        # update the position of the swing foot relative to stance control frame
        p_stance_lb = p_stance - self.tol_feet
        p_stance_ub = p_stance + self.tol_feet
        p_swing_lb = p_swing - self.tol_feet
        p_swing_ub = p_swing + self.tol_feet
        self.p_stance_foot_cons.evaluator().UpdateLowerBound(p_stance_lb)
        self.p_stance_foot_cons.evaluator().UpdateUpperBound(p_stance_ub)
        self.p_swing_foot_cons.evaluator().UpdateLowerBound(p_swing_lb)
        self.p_swing_foot_cons.evaluator().UpdateUpperBound(p_swing_ub)

        # solve the IK problem
        intial_guess = self.plant.GetPositions(self.plant_context)
        self.ik.prog().SetInitialGuess(self.ik.q(), intial_guess)
        res = SnoptSolver().Solve(self.ik.prog())

        return res

    # ---------------------------------------------------------------------------------------- #
    def CalcOutput(self, context, output):

        # set our interal model to match the state estimate
        x_hat = self.EvalVectorInput(context, 0).get_value()
        self.plant.SetPositionsAndVelocities(self.plant_context, x_hat)
        self.t_current = context.get_time()

        print("\n *************************************** \n")
        print("time: ", self.t_current) 

        # solve the inverse kinematics problem
        c_W = np.array([-0., -0.1, 0.4]).reshape(3, 1)
        c_st = np.array([0., 0.2, 0.1]).reshape(3, 1)
        r = 0.01
        rps = 0.5
        omega = -2*np.pi*rps
        # p_right_des = np.array([[c[0] + r*np.cos(omega*self.t_current)], [c[1]], [c[2] + r*np.sin(omega*self.t_current)]])
        # p_left_des = np.array([[c[0] - r*np.cos(omega*self.t_current)], [c[1]], [c[2] - r*np.sin(omega*self.t_current)]])
        p_stance_des = c_W
        p_swing_des = c_st

        # draw spehere at desired foot positions
        # self.meshcat.SetTransform("p_right", RigidTransform(p_right_des), self.t_current)
        # self.meshcat.SetTransform("p_left", RigidTransform(p_left_des), self.t_current)
        self.meshcat.SetTransform("p_stance", RigidTransform(p_stance_des), self.t_current)
        self.meshcat.SetTransform("p_swing", RigidTransform(p_swing_des + p_stance_des), self.t_current)

        # solve the IK problem
        # res = self.DoInverseKinematics(p_right_des, 
        #                                p_left_des)
        res = self.DoInverseKinematics(p_stance_des, p_swing_des)
        
        # extract the IK solution
        q_ik = res.GetSolution(self.ik.q())
    
        # compute the nominal state
        q_des = np.array([q_ik[0], q_ik[1], q_ik[2], q_ik[3], q_ik[4],  # left leg:  hip_yaw, hip_roll, hip_pitch, knee, ankle 
                          q_ik[5], q_ik[6], q_ik[7], q_ik[8], q_ik[9]]) # right leg: hip_yaw, hip_roll, hip_pitch, knee, ankle
        v_des = np.zeros(self.plant.num_actuators())
        x_des = np.block([q_des, v_des])

        output.SetFromVector(x_des)

######################################################################################

# simulation parameters
sim_time = 5.0
realtime_rate = 0.5

# load model
model_file = "../models/achilles_SE3_drake_ik.urdf"

# start meshcat
meshcat = StartMeshcat()

# simulation parameters
sim_hz = 800
sim_config = MultibodyPlantConfig()
sim_config.time_step = 1 / sim_hz 
sim_config.discrete_contact_approximation = "lagged"
sim_config.contact_model = "hydroelastic_with_fallback"

# Set up the Drake system diagram
builder = DiagramBuilder()
plant, scene_graph = AddMultibodyPlant(sim_config, builder)

# Add the harpy model
robot = Parser(plant).AddModels(model_file)[0]

# add gravity
plant.gravity_field().set_gravity_vector([0, 0, -9.81])

# add low level PD controllers
kp = 500
Kp = np.array([kp, kp, kp, kp, kp, kp, kp, kp, kp, kp])
kd = 10
Kd = np.array([kd, kd, kd, kd, kd, kd, kd, kd, kd, kd])
actuator_indices = [JointActuatorIndex(i) for i in range(plant.num_actuators())]
for actuator_index, Kp, Kd in zip(actuator_indices, Kp, Kd):
    plant.get_joint_actuator(actuator_index).set_controller_gains(
    PdControllerGains(p=Kp, d=Kd)
)

# finalize the plant
plant.Finalize()

# add the controller
c = Controller(model_file, meshcat)
controller = builder.AddSystem(c)

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
q0 = np.array([0, 0, 0, 0, 0,  # left leg:  hip_yaw, hip_roll, hip_pitch, knee, ankle 
               0, 0, 0, 0, 0]) # right leg: hip_yaw, hip_roll, hip_pitch, knee, ankle
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
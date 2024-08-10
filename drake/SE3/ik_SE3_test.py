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
        self.sphere = Sphere(0.025)

        self.meshcat.SetObject("p_right", self.sphere, self.red_color)
        self.meshcat.SetObject("p_left", self.sphere, self.blue_color)

        # create internal model of the robot
        self.plant = MultibodyPlant(0)
        Parser(self.plant).AddModels(model_file)
    
        # relevant frames
        self.left_foot_frame = self.plant.GetFrameByName("left_foot")
        self.right_foot_frame = self.plant.GetFrameByName("right_foot")
        self.stance_foot_frame = self.right_foot_frame
        self.swing_foot_frame = self.left_foot_frame
        
        self.T = 1.0
        self.t_phase = 0.0
        self.num_steps = 0
        self.update_foot_role_flag = True

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
        epsilon_feet = 0.005        # foot position tolerance     [m]
        self.foot_epsilon_orient = 0.0   # foot orientation tolerance  [deg]
        self.tol_feet = np.array([[epsilon_feet], [epsilon_feet], [epsilon_feet]])  # x-z only
        
        # Add foot position constraints (continuously updated)
        self.p_left_cons =  self.ik.AddPositionConstraint(self.left_foot_frame, [0, 0, 0],
                                                          self.plant.world_frame(), 
                                                          [0, 0, 0], [0, 0, 0])
        self.p_right_cons = self.ik.AddPositionConstraint(self.right_foot_frame, [0, 0, 0],
                                                          self.plant.world_frame(), 
                                                          [0, 0, 0], [0, 0, 0]) 

        # Add foot orientation constraints (removed and created at each HLIP step)
        self.r_left_cons = self.ik.AddAngleBetweenVectorsConstraint(self.left_foot_frame, [1, 0, 0],
                                                                    self.plant.world_frame(), [1, 0, 0],
                                                                    0, 0.0)
        self.r_right_cons = self.ik.AddAngleBetweenVectorsConstraint(self.right_foot_frame, [1, 0, 0],
                                                                    self.plant.world_frame(), [1, 0, 0],
                                                                    0, 0.0)

    # ---------------------------------------------------------------------------------------- #

    # update the foot role
    def update_foot_roles(self):
        
        # check if entered new step period
        if self.t_phase >= self.T:
            self.num_steps += 1
            self.update_foot_role_flag = True

        # update the foot role
        if self.update_foot_role_flag == True:

            # left foot is swing foot
            if self.num_steps %2 == 0:

                # set the last known swing foot position as the desried stance foot position
                p_stance = self.plant.CalcPointsPositions(self.plant_context,
                                                          self.stance_foot_frame, [0, 0, 0],  # NOTE: sholdnt this be swing?
                                                          self.plant.world_frame())
                R_stance = self.plant.CalcRelativeRotationMatrix(self.plant_context,
                                                                 self.plant.world_frame(),
                                                                 self.stance_foot_frame)
                _, _, yaw = RollPitchYaw(R_stance).vector()
                self.p_control_stance_W = np.array([p_stance[0], p_stance[1], p_stance[2]])
                self.R_control_stance_W = RotationMatrix(RollPitchYaw(0, 0, yaw)).matrix()

                # remove the old foot orientation constraints
                self.ik.prog().RemoveConstraint(self.r_left_cons)
                self.ik.prog().RemoveConstraint(self.r_right_cons)

                # add the new foot orientation constraints
                self.r_left_cons = self.ik.AddAngleBetweenVectorsConstraint(self.left_foot_frame, [1, 0, 0],
                                                                            self.plant.world_frame(), self.R_control_stance_W @ [1, 0, 0],
                                                                            0, self.foot_epsilon_orient * np.pi / 180)
                self.r_right_cons = self.ik.AddAngleBetweenVectorsConstraint(self.right_foot_frame, [1, 0, 0],
                                                                             self.plant.world_frame(), self.R_control_stance_W @ [1, 0, 0],
                                                                             0, self.foot_epsilon_orient * np.pi / 180)                

                # update the stance foot position constraints
                self.p_right_nom = np.array([[0], [-0.1], [0.2]])
                p_right_lb = self.p_right_nom - self.tol_feet
                p_right_ub = self.p_right_nom + self.tol_feet
                self.p_right_cons.evaluator().UpdateLowerBound(p_right_lb)
                self.p_right_cons.evaluator().UpdateUpperBound(p_right_ub)
                self.meshcat.SetTransform("p_right", RigidTransform(self.p_right_nom), self.t_current)

                # switch the roles of the feet
                self.swing_foot_frame = self.left_foot_frame
                self.stance_foot_frame = self.right_foot_frame

            # right foot is swing foot
            else:

                # set the last known swing foot position as the desried stance foot position
                p_stance = self.plant.CalcPointsPositions(self.plant_context,
                                                            self.stance_foot_frame, [0, 0, 0], # NOTE: sholdnt this be swing?
                                                            self.plant.world_frame())
                R_stance = self.plant.CalcRelativeRotationMatrix(self.plant_context,
                                                                self.plant.world_frame(),
                                                                self.stance_foot_frame)
                _, _, yaw = RollPitchYaw(R_stance).vector()
                self.p_control_stance_W = np.array([p_stance[0], p_stance[1], p_stance[2]])
                self.R_control_stance_W = RotationMatrix(RollPitchYaw(0, 0.0, yaw)).matrix()

                # remove the old foot orientation constraints
                self.ik.prog().RemoveConstraint(self.r_left_cons)
                self.ik.prog().RemoveConstraint(self.r_right_cons)

                # add the new foot orientation constraints
                self.r_left_cons = self.ik.AddAngleBetweenVectorsConstraint(self.left_foot_frame, [1, 0, 0],
                                                                            self.plant.world_frame(), [np.cos(yaw), np.sin(yaw), 0],
                                                                            0, self.foot_epsilon_orient * np.pi / 180)
                self.r_right_cons = self.ik.AddAngleBetweenVectorsConstraint(self.right_foot_frame, [1, 0, 0],
                                                                             self.plant.world_frame(), [np.cos(yaw), np.sin(yaw), 0],
                                                                             0, self.foot_epsilon_orient * np.pi / 180)
                
                # update the stance foot position constraints
                self.p_left_nom = np.array([[0], [0.1], [0.2]])
                p_left_lb = self.p_left_nom - self.tol_feet
                p_left_ub = self.p_left_nom + self.tol_feet
                self.p_left_cons.evaluator().UpdateLowerBound(p_left_lb)
                self.p_left_cons.evaluator().UpdateUpperBound(p_left_ub)
                self.meshcat.SetTransform("p_left", RigidTransform(self.p_left_nom), self.t_current)

                # switch the roles of the feet
                self.swing_foot_frame = self.right_foot_frame
                self.stance_foot_frame = self.left_foot_frame

            # reset the foot role flag after switching
            self.update_foot_role_flag = False   

        # update the phase time
        self.t_phase = self.t_current - self.num_steps * self.T

    # given desired foot and torso positions, solve the IK problem
    def DoInverseKinematics(self, p_swing_C):

        # Update constraints on the positions of the feet
        if self.swing_foot_frame == self.left_foot_frame:
            p_left_W = self.p_right_nom + (self.R_control_stance_W @ p_swing_C).reshape(3,1)
            self.p_left_cons.evaluator().UpdateLowerBound(p_left_W - self.tol_feet)
            self.p_left_cons.evaluator().UpdateUpperBound(p_left_W + self.tol_feet)
            self.meshcat.SetTransform("p_left", RigidTransform(p_left_W), self.t_current)
            
        elif self.swing_foot_frame == self.right_foot_frame:
            p_right_W = self.p_left_nom + (self.R_control_stance_W @ p_swing_C).reshape(3,1)
            self.p_right_cons.evaluator().UpdateLowerBound(p_right_W - self.tol_feet)
            self.p_right_cons.evaluator().UpdateUpperBound(p_right_W + self.tol_feet)
            self.meshcat.SetTransform("p_right", RigidTransform(p_right_W), self.t_current)

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

        # update the foot roles
        self.update_foot_roles()
        r = 0.05
        rps = 2.0
        omega = -2*np.pi*rps
        if self.stance_foot_frame == self.right_foot_frame:
            p_swing_C_center = np.array([0.0, 0.2, 0.1])
            p_swing_C = np.array([0.0, r*np.cos(omega*self.t_phase), r*np.sin(omega*self.t_phase)]) + p_swing_C_center
        elif self.stance_foot_frame == self.left_foot_frame:
            p_swing_C_center = np.array([0.0, -0.2, 0.1])
            p_swing_C = np.array([0.0, r*np.cos(omega*self.t_phase), r*np.sin(omega*self.t_phase)]) + p_swing_C_center

        # solve the IK problem
        res = self.DoInverseKinematics(p_swing_C)
        
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
sim_time = 10.0
realtime_rate = 1.0

# load model
model_file = "../models/achilles_SE3_drake_ik.urdf"

# start meshcat
meshcat = StartMeshcat()

# simulation parameters
sim_hz = 500
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
kp = 250
Kp = np.array([kp, kp, kp, kp, kp, kp, kp, kp, kp, kp])
kd = 3
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
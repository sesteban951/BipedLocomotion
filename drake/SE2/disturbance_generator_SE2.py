import numpy as np
from pydrake.all import (LeafSystem, BasicVector, Cylinder, Rgba, 
                        RigidTransform, RotationMatrix)   


class DisturbanceGenerator(LeafSystem):
    """
    A simple Drake system that generates a fixed disturbance. 

    This disturbance is a constant force applied to the 
    `generalized_force_input_port` of a multibody plant.
    """

    def __init__(self, plant, meshcat, generalized_force, time, duration):
        """
        Construct the disturbance generator system.

        Args:
            plant: A MultibodyPlant object representing the system
            generalized_force: The constant force to apply as a disturbance
            time: The time at which to start the disturbance (float)
            duration: The duration of the disturbance (float)
        """
        LeafSystem.__init__(self)

        # state sizes
        self.nq = plant.num_positions()
        self.nv = plant.num_velocities()

        # Check that the generalized force has the correct shape
        assert generalized_force.shape[0] == self.nv

        # Store the parameters
        self.tau = generalized_force
        self.time = time
        self.duration = duration
        self.meshcat = meshcat

        self.f_base_W = np.array([self.tau[0], 0, self.tau[1]])
        self.f_norm = np.linalg.norm(self.f_base_W)

        # visual vector scaling factor
        self.visual_scaling = 0.1

        # define a cylinder primitive to represent the force
        color = Rgba(1, 0, 0, 1)
        cyl = Cylinder(0.0075, 1.0)
        self.meshcat.SetObject("force", cyl, color)
        self.meshcat.SetProperty("force", "scale", (1, 1, 1), 0.0)

        # Declare the output port
        self.sate_input_port = self.DeclareVectorInputPort("state",
                                                           BasicVector(self.nq + self.nv))
        self.DeclareVectorOutputPort("generalized_force",
                                     BasicVector(plant.num_velocities()),
                                     self.CalcOutput)

    def rotation_matrix_from_points(self, p1, p2):
        """
        Returns a 3D rotation matrix that aligns with the line segment connecting two points.
        
        Parameters:
        p1: numpy array of shape (3, 1) - starting point
        p2: numpy array of shape (3, 1) - ending point
        
        Returns:
        R: 3x3 numpy array representing the rotation matrix
        """
        # Calculate the direction vector between p1 and p2
        v = p2 - p1
        # Normalize the vector to get the unit vector along the line segment
        v_hat = v / np.linalg.norm(v)

        # Choose a reference vector (not aligned with v_hat). We'll choose the x-axis unit vector [1, 0, 0].
        # If v_hat is aligned with the x-axis, choose another arbitrary vector like [0, 1, 0].
        if np.allclose(v_hat.flatten(), np.array([1, 0, 0])) or np.allclose(v_hat.flatten(), np.array([-1, 0, 0])):  # if aligned with x-axis
            e_ref = np.array([0, 1, 0])  # pick y-axis as reference
        else:
            e_ref = np.array([1, 0, 0])  # pick x-axis as reference

        # Compute the first orthogonal vector by taking the cross product
        u1 = np.cross(v_hat.flatten(), e_ref)
        u1 = u1 / np.linalg.norm(u1)  # normalize the vector to get unit vector u1

        # Compute the second orthogonal vector as the cross product of v_hat and u1
        u2 = np.cross(v_hat.flatten(), u1)
        u2 = u2 / np.linalg.norm(u2)  # normalize the vector to get unit vector u2

        # Construct the rotation matrix using u1, u2, and v_hat
        R = np.column_stack((u1, u2, v_hat.flatten()))

        return R

    # draw the force in meshcat
    def DrawForce(self, do_draw_force, state, t):
        
        # get the current postion of the base
        if do_draw_force == True:
            q = state[:self.nq]

            # get the base adn force positions
            p_base_W = np.array([q[0], 0, q[1]])
            f_base_W_scaled = self.f_base_W * self.visual_scaling
            
            # compute the position of the force for visualization
            p_cyl = (f_base_W_scaled + 2 * p_base_W) / 2
            R_cyl = self.rotation_matrix_from_points(np.array([0,0,0]), self.f_base_W)
            print(R_cyl)
            transform = RigidTransform(RotationMatrix(R_cyl), p_cyl)

            print(self.f_norm)

            # set the force position and scale
            self.meshcat.SetProperty("force", "scale", (1, 1, self.f_norm * self.visual_scaling), t)
            self.meshcat.SetTransform("force", transform, t)
        else:
            self.meshcat.SetTransform("force", RigidTransform(RotationMatrix(), np.array([100,0,0])), t)

    def CalcOutput(self, context, output):
        """
        Calculate the output of the disturbance generator.

        Args:
            context: The context of the system
            output: The output of the system
        """
        # evaluate the current state
        state = self.sate_input_port.Eval(context)

        # boolean to see if we apply the force
        do_apply_force = (context.get_time() >= self.time and context.get_time() < self.time + self.duration)
        
        # draw the force in meshcat
        self.DrawForce(do_apply_force, state,  context.get_time())

        # set generalized force output
        if do_apply_force == True:
            output.SetFromVector(self.tau)
        else:
            output.SetFromVector(np.zeros_like(self.tau))
        
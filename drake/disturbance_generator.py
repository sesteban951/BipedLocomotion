import numpy as np
from pydrake.all import LeafSystem, BasicVector


class DisturbanceGenerator(LeafSystem):
    """
    A simple Drake system that generates a fixed disturbance. 

    This disturbance is a constant force applied to the 
    `generalized_force_input_port` of a multibody plant.
    """

    def __init__(self, plant, generalized_force, time, duration):
        """
        Construct the disturbance generator system.

        Args:
            plant: A MultibodyPlant object representing the system
            generalized_force: The constant force to apply as a disturbance
            time: The time at which to start the disturbance (float)
            duration: The duration of the disturbance (float)
        """
        LeafSystem.__init__(self)

        assert generalized_force.shape[0] == plant.num_velocities()
        self.tau = generalized_force
        self.time = time
        self.duration = duration

        # Declare the output port
        self.DeclareVectorOutputPort("generalized_force",
                                     BasicVector(plant.num_velocities()),
                                     self.CalcOutput)
        
    def CalcOutput(self, context, output):
        """
        Calculate the output of the disturbance generator.

        Args:
            context: The context of the system
            output: The output of the system
        """
        if context.get_time() >= self.time and context.get_time() < self.time + self.duration:
            output.SetFromVector(self.tau)
        else:
            output.SetFromVector(np.zeros_like(self.tau))
        
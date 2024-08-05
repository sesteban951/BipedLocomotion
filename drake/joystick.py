import numpy as np
import pygame
from pydrake.all import LeafSystem, BasicVector

class GamepadCommand(LeafSystem):
    """
    A system that reads commands from a gamepad controller conencted to meshcat.
    """
    def __init__(self):
        """
        Construct the gamepad command system.

        Args:
            meshcat: The meshcat visualizer object.
        """
        LeafSystem.__init__(self)

        # set an output port for the gamepad commands
        self.DeclareVectorOutputPort("gamepad_command", BasicVector(4), self.CalcOutput)

        # pygame init
        pygame.init()
        self.clock = pygame.time.Clock()
        self.keepPlaying = True
        
        # look for plugged in joysticks
        self.joysticks = []
        self.joysticks.append(pygame.joystick.Joystick(0))
        self.joysticks[0].init()  # just initialize the first joystick
        self.detected_joystick = None

        # print message for number of joysticks found
        if len(self.joysticks) == 0:
            print("No joysticks connected.")
            self.detected_joystick = False
        else:
            print("Found ", pygame.joystick.get_count(), " joysticks")
            self.detected_joystick = True

    def CalcOutput(self, context, output):
        """
        Output a desired velocity command from the gamepad.
        """
    
        # just set the output to zero if no joystick is detected
        if self.detected_joystick == False:
            # print("Gamepad not connected, sending zero commands.")
            output.SetFromVector(np.zeros(3))
        else:

            # Update internal state of Pygame
            pygame.event.pump()  

            # map the joystick axes (XBOX One Controller)
            LS_x =  self.joysticks[0].get_axis(0)  # Left Stick X-direction
            LS_y = -self.joysticks[0].get_axis(1)  # Left Stick Y-direction
            RS_x =  self.joysticks[0].get_axis(3)  # Right Stick X-direction
            A = self.joysticks[0].get_button(0)    # A button

            output[0] = LS_x  # x velocity command
            output[1] = LS_y  # y velocity command
            output[2] = RS_x  # z angular velocity command
            output[3] = A     # misc. boolean button command

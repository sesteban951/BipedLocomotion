import numpy as np
import pygame
from pydrake.all import LeafSystem, BasicVector

class GamepadCommand(LeafSystem):
    """
    Args:
        - deadzone: float in [0,1], deadzone for the joystick axes
    """
    def __init__(self, deadzone):
        
        LeafSystem.__init__(self)

        # set an output port for the gamepad commands
        self.DeclareVectorOutputPort("gamepad_command", BasicVector(5), self.CalcOutput)

        # set the deadzone value
        self.deadzone = deadzone
        assert 0 <= self.deadzone <= 1, "Deadzone must be in [0,1]"

        # pygame init
        pygame.init()
        self.clock = pygame.time.Clock()
        self.keepPlaying = True
        
        # look for plugged in joysticks
        if pygame.joystick.get_count() == 0:
            print("No joysticks connected.")
            self.detected_joystick = False
        else:
            print("Found ", pygame.joystick.get_count(), " joysticks")
            self.detected_joystick = True
            self.joysticks = []
            self.joysticks.append(pygame.joystick.Joystick(0))
            self.joysticks[0].init()  # just initialize the first joystick

    def CalcOutput(self, context, output):
        """
        Output a desired velocity command from the gamepad.
        """
    
        # just set the output to zero if no joystick is detected
        if self.detected_joystick == False:
            # print("Gamepad not connected, sending zero commands.")
            output.SetFromVector(np.zeros(5))
        else:

            # Update internal state of Pygame
            pygame.event.pump()  

            # map the joystick axes (XBOX One Controller)
            LS_x = -self.joysticks[0].get_axis(0)             # Left Stick X-direction  (left = +1, right = -1)
            LS_y = -self.joysticks[0].get_axis(1)             # Left Stick Y-direction  (up = -1, down = +1)
            RS_x = -self.joysticks[0].get_axis(3)             # Right Stick X-direction (left = -1, right = +1)
            A = self.joysticks[0].get_button(0)               # A button (unpressed = 0, pressed = 1)
            # RT = 0.5 * (self.joysticks[0].get_axis(5) + 1.0)  # Right Trigger (unpressed = 0, pressed = 1)
            RT = 0.0

            # set deadzone to combat the stick drift
            if abs(LS_x) < self.deadzone:
                LS_x = 0.0
            if abs(LS_y) < self.deadzone:
                LS_y = 0.0
            if abs(RS_x) < self.deadzone:
                RS_x = 0.0
            if abs(RT) < self.deadzone:
                RT = 0.0

            output[0] = LS_x  # x velocity command
            output[1] = LS_y  # y velocity command
            output[2] = RS_x  # z angular velocity command
            output[3] = A     # misc. boolean button command
            output[4] = RT    # right trigger command

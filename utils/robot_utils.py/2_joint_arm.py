import pygame
import numpy as np

# Initialize Pygame
pygame.init()

# Screen dimensions and colors
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GROUND_Y = 500  # Ground level

# Arm class with two joints
class TwoJointArm:
    def __init__(self, base_position, link_lengths, ground_y):
        self.base_position = base_position  # Base position of the arm
        self.link_lengths = link_lengths    # Length of each arm link
        self.joint_angles = [0, 0]          # Initial joint angles (in radians)
        self.ground_y = ground_y            # Ground level for collision detection

    def forward_kinematics(self):
        """Calculate the end positions of each link based on joint angles."""
        x1 = self.base_position[0] + self.link_lengths[0] * np.cos(self.joint_angles[0])
        y1 = self.base_position[1] + self.link_lengths[0] * np.sin(self.joint_angles[0])

        x2 = x1 + self.link_lengths[1] * np.cos(self.joint_angles[0] + self.joint_angles[1])
        y2 = y1 + self.link_lengths[1] * np.sin(self.joint_angles[0] + self.joint_angles[1])

        return (x1, y1), (x2, y2)

    def control(self, key_input):
        """Update joint angles based on keyboard input."""
        if key_input[pygame.K_LEFT]:
            self.joint_angles[0] -= 0.05
        if key_input[pygame.K_RIGHT]:
            self.joint_angles[0] += 0.05
        if key_input[pygame.K_DOWN]:
            self.joint_angles[1] -= 0.05
        if key_input[pygame.K_UP]:
            self.joint_angles[1] += 0.05

    def check_collision(self):
        """Check for self-collision and ground collision."""
        joint1_pos, end_effector_pos = self.forward_kinematics()

        # Check ground collision (y-position of end effector and first joint)
        ground_collision = joint1_pos[1] > self.ground_y or end_effector_pos[1] > self.ground_y

        # Check self-collision (in this case, the two-link arm cannot self-collide in 2D)
        # To detect self-collision in more complex arms, we could add other checks

        return ground_collision  # Return True if ground collision detected

    def draw(self, screen):
        """Draw the arm on the Pygame screen."""
        joint1_pos, end_effector_pos = self.forward_kinematics()

        # Draw the base, links, and joints
        pygame.draw.circle(screen, BLACK, self.base_position, 5)
        pygame.draw.line(screen, BLUE, self.base_position, joint1_pos, 5)
        pygame.draw.line(screen, GREEN, joint1_pos, end_effector_pos, 5)
        pygame.draw.circle(screen, RED, (int(joint1_pos[0]), int(joint1_pos[1])), 5)
        pygame.draw.circle(screen, RED, (int(end_effector_pos[0]), int(end_effector_pos[1])), 5)

# Environment class with ground and arm
class SimulationEnvironment:
    def __init__(self, screen_width, screen_height, ground_y):
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("2-Joint Robot Arm with Ground and Collision Detection")
        self.clock = pygame.time.Clock()
        self.ground_y = ground_y  # Ground level
        self.arm = TwoJointArm(base_position=(400, ground_y), link_lengths=[100, 100], ground_y=self.ground_y)

    def draw_ground(self):
        """Draw the ground as a black rectangle across the bottom of the screen."""
        pygame.draw.rect(self.screen, BLACK, (0, self.ground_y, SCREEN_WIDTH, SCREEN_HEIGHT - self.ground_y))

    def run(self):
        """Run the simulation environment."""
        running = True
        while running:
            self.screen.fill(WHITE)  # Clear screen with white background

            # Event handling to exit simulation
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Handle arm control
            keys = pygame.key.get_pressed()
            self.arm.control(keys)

            # Check for collision
            if self.arm.check_collision():
                print("Collision detected!")

            # Draw elements
            self.draw_ground()  # Draw the ground
            self.arm.draw(self.screen)  # Draw the arm

            pygame.display.flip()  # Update the screen
            self.clock.tick(30)  # Limit to 30 FPS

        pygame.quit()

# Run the simulation
if __name__ == '__main__':
    env = SimulationEnvironment(SCREEN_WIDTH, SCREEN_HEIGHT, GROUND_Y)
    env.run()

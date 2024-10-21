"""
In this experiment, we demonstrate forward kinematics for a simple 2-joint robotic arm using 
both the space and body frames. We calculate the end-effector position using the product of 
exponentials formula for forward kinematics and visualize the arm in the XY plane.
"""

import numpy as np
import matplotlib.pyplot as plt

from basic_algos.forward_kinematics.product_of_exponentials_formula import (
    compute_screw_axis, 
    forward_kinematics_in_space, 
    forward_kinematics_in_body
)

def plot_robot_arm(base_pos, link1_end, link2_end):
    """
    Visualizes the robotic arm in the XY plane.
    
    Parameters:
    - base_pos: The position of the base of the robot.
    - link1_end: The position of the end of the first link.
    - link2_end: The position of the end effector (end of the second link).
    """
    fig, ax = plt.subplots()

    # Plot the base and the links
    ax.plot([base_pos[0], link1_end[0]], [base_pos[1], link1_end[1]], 'b-', label='Link 1', linewidth=4)
    ax.plot([link1_end[0], link2_end[0]], [link1_end[1], link2_end[1]], 'r-', label='Link 2', linewidth=4)

    # Mark the base, joint, and end effector
    ax.plot(base_pos[0], base_pos[1], 'go', markersize=10, label='Base')
    ax.plot(link1_end[0], link1_end[1], 'bo', markersize=10, label='Joint 1')
    ax.plot(link2_end[0], link2_end[1], 'ro', markersize=10, label='End Effector')

    # Set up the plot
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_aspect('equal')
    ax.grid(True)
    ax.legend()
    plt.title('2-Joint Manipulator in XY Plane')
    plt.show()

if __name__ == "__main__":
    """
    In this experiment, we demonstrate forward kinematics for a simple 2-joint robotic arm using 
    both the space and body frames. We calculate the end-effector position using the product of 
    exponentials formula for forward kinematics and visualize the arm in the XY plane.
    """

    # Define link lengths and joint angles
    L1 = 2  # Length of the first link
    L2 = 1  # Length of the second link
    theta1 = np.pi / 4  # Joint 1 angle (45 degrees)
    theta2 = np.pi / 4  # Joint 2 angle (45 degrees)

    # Define joint screw axes in the space frame
    S1_space = compute_screw_axis(np.array([0, 0, 0]), np.array([0, 0, 1]))  # Joint 1 screw axis in space frame
    S2_space = compute_screw_axis(np.array([L1, 0, 0]), np.array([0, 0, 1]))  # Joint 2 screw axis in space frame
    screws_space = [S1_space, S2_space]  # List of joint screw axes in space frame

    # Define joint screw axes in the body frame
    S1_body = compute_screw_axis(np.array([0, 0, 0]), np.array([0, 0, 1]))  # Joint 1 screw axis in body frame (same as space)
    S2_body = compute_screw_axis(np.array([L2, 0, 0]), np.array([0, 0, 1]))  # Joint 2 screw axis in body frame
    screws_body = [S1_body, S2_body]  # List of joint screw axes in body frame

    thetas = [theta1, theta2]  # List of joint angles

    # Home configuration of the end-effector (zero position)
    M = np.array([
        [1, 0, 0, L1 + L2],  # Total length along the x-axis
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

    # Forward kinematics in space frame
    T_end_effector_space = forward_kinematics_in_space(M, screws_space, thetas)
    print("End Effector Transformation Matrix (Space Frame):\n", T_end_effector_space)

    # Extract end-effector position in space frame
    end_effector_pos_space = T_end_effector_space[:2, 3]
    print("End Effector Position (Space Frame) (x, y):", end_effector_pos_space)

    # Forward kinematics in body frame
    T_end_effector_body = forward_kinematics_in_body(M, screws_body, thetas)
    print("End Effector Transformation Matrix (Body Frame):\n", T_end_effector_body)

    # Extract end-effector position in body frame
    end_effector_pos_body = T_end_effector_body[:2, 3]
    print("End Effector Position (Body Frame) (x, y):", end_effector_pos_body)

    # Visualization
    # Link 1 endpoint is the transformation after applying theta1
    link1_end = np.array([L1 * np.cos(theta1), L1 * np.sin(theta1)])

    # End effector position is calculated from the space frame result
    end_effector_pos = end_effector_pos_space

    # Plot the robotic arm
    plot_robot_arm(base_pos=np.array([0, 0]), link1_end=link1_end, link2_end=end_effector_pos)
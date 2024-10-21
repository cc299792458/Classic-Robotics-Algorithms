"""
In this experiment, we demonstrate the usage of the from_euler and to_euler methods, 
and show the convenience of visualizing rotation using Euler angles.
"""

import numpy as np
import matplotlib.pyplot as plt

from utils.misc_utils import set_seed
from basic_algos.rigid_body_motions.rotation_matrix import RotationMatrix

def plot_axes(ax, R, label=""):
    """
    Plot the 3D axes given a rotation matrix R. The axes are colored as X=red, Y=green, Z=blue.
    """
    origin = np.zeros(3)
    X = R @ np.array([1, 0, 0])
    Y = R @ np.array([0, 1, 0])
    Z = R @ np.array([0, 0, 1])
    
    ax.quiver(*origin, *X, color='r', label=f'{label} X-axis')
    ax.quiver(*origin, *Y, color='g', label=f'{label} Y-axis')
    ax.quiver(*origin, *Z, color='b', label=f'{label} Z-axis')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])

if __name__ == '__main__':
    set_seed()
    # 1. Define Euler angles (ZYX order: yaw, pitch, roll)
    original_yaw = np.pi / 2  # 90 degrees
    original_pitch = np.pi / 2  # 90 degrees
    original_roll = np.pi / 2  # 90 degrees

    print(f"Original Euler angles (yaw, pitch, roll):")
    print(f"Yaw (Z-axis rotation): {np.degrees(original_yaw)} degrees")
    print(f"Pitch (Y-axis rotation): {np.degrees(original_pitch)} degrees")
    print(f"Roll (X-axis rotation): {np.degrees(original_roll)} degrees\n")

    # 2. Create intermediate rotation matrices
    # Rotation around Z-axis (yaw)
    Rz = np.array([[np.cos(original_yaw), -np.sin(original_yaw), 0],
                   [np.sin(original_yaw),  np.cos(original_yaw), 0],
                   [0,                    0,                     1]])

    # Rotation around Y-axis (pitch)
    Ry = np.array([[np.cos(original_pitch), 0, np.sin(original_pitch)],
                   [0,                     1, 0],
                   [-np.sin(original_pitch), 0, np.cos(original_pitch)]])

    # Rotation around X-axis (roll)
    Rx = np.array([[1, 0,               0],
                   [0, np.cos(original_roll), -np.sin(original_roll)],
                   [0, np.sin(original_roll),  np.cos(original_roll)]])

    # 3. Plot intermediate steps and final result
    fig = plt.figure(figsize=(16, 4))

    # Plot original axes (identity rotation matrix)
    ax1 = fig.add_subplot(141, projection='3d')
    plot_axes(ax1, np.eye(3), label="Original")
    ax1.set_title("Original Axes")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Plot after yaw (Z-axis rotation)
    ax2 = fig.add_subplot(142, projection='3d')
    plot_axes(ax2, Rz, label="After Z Rotation (Yaw)")
    ax2.set_title("After Z Rotation (Yaw)")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Plot after pitch (Y-axis rotation)
    ax3 = fig.add_subplot(143, projection='3d')
    R_zy = np.dot(Rz, Ry)
    plot_axes(ax3, R_zy, label="After Y Rotation (Pitch)")
    ax3.set_title("After Y Rotation (Pitch)")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    # Plot after roll (X-axis rotation)
    ax4 = fig.add_subplot(144, projection='3d')
    R_zyx = np.dot(R_zy, Rx)  # Complete ZYX rotation (Yaw -> Pitch -> Roll)
    plot_axes(ax4, R_zyx, label="After X Rotation (Roll)")
    ax4.set_title("After X Rotation (Roll)")
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')

    plt.show()

    # 5. Convert the final rotation matrix back to Euler angles
    rotation_matrix = RotationMatrix(R_zyx)
    recovered_roll, recovered_pitch, recovered_yaw = rotation_matrix.to_euler()

    print("Recovered Euler angles from the rotation matrix:")
    print(f"Recovered Yaw (Z-axis rotation): {np.degrees(recovered_yaw)} degrees")
    print(f"Recovered Pitch (Y-axis rotation): {np.degrees(recovered_pitch)} degrees")
    print(f"Recovered Roll (X-axis rotation): {np.degrees(recovered_roll)} degrees\n")

    # 6. Check if the recovered Euler angles match the original ones
    if (np.isclose(original_yaw, recovered_yaw) and
        np.isclose(original_pitch, recovered_pitch) and
        np.isclose(original_roll, recovered_roll)):
        print("Success: The recovered Euler angles match the original ones!")
    else:
        print("Error: The recovered Euler angles do not match the original ones.")

"""
In this experiment, we start with a quaternion, convert it to a rotation matrix, then to Euler angles which we use to visualize the rotations, 
and finally convert the Euler angles back to a rotation matrix and to a quaternion, comparing the final quaternion with the original.
"""

import os
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
    log_dir = os.path.dirname(os.path.abspath(__file__))
    # 1. Define quaternions for 90-degree rotations around Z, Y, and X axes
    qz = [0.7071, 0.0, 0.0, 0.7071]  # 90-degree rotation around Z axis
    qy = [0.7071, 0.0, 0.7071, 0.0]  # 90-degree rotation around Y axis
    qx = [0.7071, 0.7071, 0.0, 0.0]  # 90-degree rotation around X axis

    # 2. Combine the quaternions (ZYX order: rotate around Z, then Y, then X)
    rotation_matrix_z = RotationMatrix()
    rotation_matrix_z.from_quaternion(qz)

    rotation_matrix_y = RotationMatrix()
    rotation_matrix_y.from_quaternion(qy)

    rotation_matrix_x = RotationMatrix()
    rotation_matrix_x.from_quaternion(qx)

    # Combine the rotation matrices by multiplying them (ZYX order)
    rotation_matrix_combined = rotation_matrix_z * rotation_matrix_y * rotation_matrix_x

    print(f"\nCombined rotation matrix from 90-degree rotations around Z, Y, and X axes:\n{rotation_matrix_combined.matrix}")

    # 3. Convert the combined rotation matrix to Euler angles
    roll, pitch, yaw = rotation_matrix_combined.to_euler()
    print(f"\nEuler angles (roll, pitch, yaw) in radians: ({roll}, {pitch}, {yaw})")
    print(f"Euler angles (roll, pitch, yaw) in degrees: ({np.degrees(roll)}, {np.degrees(pitch)}, {np.degrees(yaw)})\n")

    # 4. Visualization
    fig = plt.figure(figsize=(16, 4))

    # Plot original axes (identity rotation matrix)
    ax1 = fig.add_subplot(141, projection='3d')
    plot_axes(ax1, np.eye(3), label="Original")
    ax1.set_title("Original Axes")
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')

    # Plot after yaw (Z-axis rotation)
    Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                   [np.sin(yaw),  np.cos(yaw), 0],
                   [0,                    0,   1]])
    ax2 = fig.add_subplot(142, projection='3d')
    plot_axes(ax2, Rz, label="After Z Rotation (Yaw)")
    ax2.set_title("After Z Rotation (Yaw)")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')

    # Plot after pitch (Y-axis rotation)
    Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                   [0,             1, 0],
                   [-np.sin(pitch), 0, np.cos(pitch)]])
    ax3 = fig.add_subplot(143, projection='3d')
    R_zy = np.dot(Rz, Ry)
    plot_axes(ax3, R_zy, label="After Y Rotation (Pitch)")
    ax3.set_title("After Y Rotation (Pitch)")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')

    # Plot after roll (X-axis rotation)
    Rx = np.array([[1, 0,               0],
                   [0, np.cos(roll), -np.sin(roll)],
                   [0, np.sin(roll),  np.cos(roll)]])
    ax4 = fig.add_subplot(144, projection='3d')
    R_zyx = np.dot(R_zy, Rx)  # Complete ZYX rotation (Yaw -> Pitch -> Roll)
    plot_axes(ax4, R_zyx, label="After X Rotation (Roll)")
    ax4.set_title("After X Rotation (Roll)")
    ax4.set_xlabel('X')
    ax4.set_ylabel('Y')
    ax4.set_zlabel('Z')

    # Add a title to the entire figure
    plt.suptitle('Quaternion Rotation Visualization: ZYX 90-degree Rotations', fontsize=16)

    plt.savefig(os.path.join(log_dir, 'quaternion_rotation_visualization.png'))

    plt.show()

    # 5. Convert Euler angles back to rotation matrix
    rotation_matrix_back = RotationMatrix()
    rotation_matrix_back.from_euler(roll, pitch, yaw)

    # 6. Convert the new rotation matrix back to a quaternion
    recovered_quaternion = rotation_matrix_back.to_quaternion()
    print(f"\nRecovered Quaternion (qw, qx, qy, qz): {recovered_quaternion}")
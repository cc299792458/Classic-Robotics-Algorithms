"""
    In this experiment, we demonstrate that the rotation matrix has three main uses:
    1. Representing an orientation.
    2. Changing the reference frame in which a vector or a frame is represented.
    3. Rotating a vector or a frame within its original reference frame.
"""

import numpy as np

from basic_algos.rigid_body_motions.rotation_matrix import RotationMatrix

if __name__ == '__main__':
    # 1. Representing the orientation of frame A, B and C relative to frame A
    R_aa = RotationMatrix()  # Identity matrix: frame A relative to itself (no rotation)
    
    R_ab = RotationMatrix([[0.0, -1.0, 0.0],  # Rotation matrix representing frame B relative to frame A
                           [1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0]])
    
    R_ac = RotationMatrix([[0.0, -1.0, 0.0],  # Rotation matrix representing frame C relative to frame A
                           [0.0, 0.0, -1.0],
                           [1.0, 0.0, 0.0]])
    
    # Define a point in frame A
    p_a = np.array([1.0, 1.0, 0.0])  

    # 2. Changing the reference frame
    # Compute the inverse of R_ab to represent frame A relative to frame B
    R_ba = R_ab.inverse()  # R_ba: Rotation matrix representing frame A relative to frame B
    
    # Change the reference frame of frame C from frame A to frame B
    R_bc = R_ba * R_ac

    # Change the reference frame of the point from frame A to frame B
    p_b = R_ba * p_a

    # 3. Rotating a vector or frame within its original reference frame
    # Define a 90-degree rotation about the X-axis
    theta = np.pi / 2
    R_x_90 = RotationMatrix([[1.0, 0.0, 0.0],  # Rotation matrix for 90 degrees about X-axis
                             [0.0, np.cos(theta), -np.sin(theta)],
                             [0.0, np.sin(theta), np.cos(theta)]])
    
    # Rotate frame B, where the rotation is expressed in frame A's reference frame
    R_ab1 = R_x_90 * R_ab  # Rotate frame B using a 90-degree rotation about X in frame A
    
    # Rotate frame B, where the rotation is expressed in frame B's own reference frame
    R_ab2 = R_ab * R_x_90  # Rotate frame B in its own reference frame (B's local rotation)
    
    # Output results
    print("Original rotation matrix R_ab (representing frame B relative to frame A):\n", R_ab)
    print("Transformed frame C relataive to frame B (from frame A):\n", R_bc)
    print("Transformed point in frame B (from frame A):\n", p_b)
    print("R_ab after first rotation (R_ab1), rotating B in frame A:\n", R_ab1)
    print("R_ab after second rotation (R_ab2), rotating B in its own frame:\n", R_ab2)

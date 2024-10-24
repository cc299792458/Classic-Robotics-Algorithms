"""
In this experiment, we demonstrate that the homogeneous transformation matrix has three main uses:
1. Representing a rigid body's pose (position and orientation).
2. Changing the reference frame in which a vector or frame is represented.
3. Rotating and translating a vector or a frame within its original reference frame.

We also perform an additional experiment to demonstrate the transformations between a homogeneous transformation matrix (SE(3)), 
its corresponding se(3) matrix, and a twist vector.
"""

import numpy as np

from utils.misc_utils import set_seed
from basic_algos.rigid_body_motions.rotation_matrix import RotationMatrix
from basic_algos.rigid_body_motions.homogeneous_transformation_matrix import HomogeneousTransformationMatrix

if __name__ == '__main__':
    set_seed()
    # 1. Representing the pose of frame A, B, and C relative to frame A
    T_aa = HomogeneousTransformationMatrix()  # Identity matrix: frame A relative to itself (no translation or rotation)
    
    T_ab = HomogeneousTransformationMatrix(R=RotationMatrix([[0.0, 0.0, 1.0],  # Rotation matrix representing frame B relative to frame A
                                                            [0.0, -1.0, 0.0],
                                                            [1.0, 0.0, 0.0]]),
                                           p=np.array([0.0, -2.0, 0.0]))  # Translation from frame A to frame B
    
    T_ac = HomogeneousTransformationMatrix(R=RotationMatrix([[-1.0, 0.0, 0.0],  # Rotation matrix representing frame C relative to frame A
                                                            [0.0, 0.0, 1.0],
                                                            [0.0, 1.0, 0.0]]),
                                           p=np.array([-1.0, 1.0, 0.0]))  # Translation from frame A to frame C
    
    # Define a point in frame A
    p_a = np.array([1.0, 1.0, 0.0])  # Point representation in frame A (x, y, z)

    # 2. Changing the reference frame
    # Compute the inverse of T_ab to represent frame A relative to frame B
    T_ba = T_ab.inverse()  # T_ba: Transformation matrix representing frame A relative to frame B
    
    # Change the reference frame of frame C from frame A to frame B
    T_bc = T_ba * T_ac  # Transformation matrix from frame B to frame C through A

    # Change the reference frame of the point from frame A to frame B
    p_b = T_ba * p_a  # Transform point from frame A to frame B
    
    # 3. Rotating and translating a frame within its original reference frame
    # Define a transformation matrix
    T = HomogeneousTransformationMatrix(R=RotationMatrix([[0.0, -1.0, 0.0],
                                                        [1.0, 0.0, 0.0],
                                                        [0.0, 0.0, 1.0]]),
                                        p=np.array([0.0, 2.0, 0.0]))
    
    # Rotate and translate frame B, where the rotation is expressed in frame A's reference frame
    T_ab1 = T * T_ab  # Rotate and translate frame B in frame A
    
    # Rotate and translate frame B, where the rotation is expressed in frame B's own reference frame
    T_ab2 = T_ab * T  # Translate and rotate frame B in its own reference frame (B's local transformation)
    
    # Output results
    print("Original transformation matrix T_ab (representing frame B relative to frame A):\n", T_ab)
    print("Transformed frame C relative to frame B (from frame A):\n", T_bc)
    print("Transformed point in frame B (from frame A):\n", p_b)
    print("T_ab after first transformation (T_ab1), transforming B in frame A:\n", T_ab1)
    print("T_ab after second transformation (T_ab2), transforming B in its own frame:\n", T_ab2)


    # -------- Additional Part: se3mat and twist demonstration --------
    # Define a simple twist: [angular velocity (omega_x, omega_y, omega_z), linear velocity (v_x, v_y, v_z)]
    twist = np.array([0.0, 0.0, np.pi / 2, 0.0, 1.0, 0.0])  # A simple 90-degree rotation around z-axis and translation along y-axis
    
    # Convert twist to SE(3) using from_twist
    T_twist = HomogeneousTransformationMatrix()
    T_twist.from_twist(twist)

    # Output SE(3) matrix from the twist
    print("\nSE(3) Matrix from Twist (90-degree rotation around z-axis + translation along y-axis):\n", T_twist.matrix)

    # Convert SE(3) matrix back to se(3) matrix
    se3mat = T_twist.to_se3mat()
    print("\nse(3) Matrix (logarithmic map of SE(3)):\n", se3mat)

    # Convert SE(3) back to twist and print the result
    recovered_twist = T_twist.to_twist()
    print("\nRecovered Twist Vector from SE(3):\n", recovered_twist)
import numpy as np

from basic_algos.rigid_body_motions.rotation_matrix import RotationMatrix

class HomogeneousTransformationMatrix:
    def __init__(self, R=None, p=None):
        """
        Initialize the homogeneous transformation matrix with:
        - R: 3x3 rotation matrix (instance of RotationMatrix)
        - p: 3D translation vector (numpy array of shape (3,))
        """
        if R is None:
            R = RotationMatrix().matrix  # Default is the identity rotation
        else:
            assert isinstance(R, RotationMatrix), "R must be an instance of RotationMatrix"
            R = R.matrix
        
        if p is None:
            p = np.zeros(3)  # Default is no translation
        else:
            assert isinstance(p, np.ndarray) and p.shape == (3,), "p must be a 3D vector"
        
        # Create the 4x4 homogeneous transformation matrix
        self.matrix = np.eye(4)
        self.matrix[:3, :3] = R  # Set the rotation part
        self.matrix[:3, 3] = p   # Set the translation part

    def inverse(self):
        """
        Return the inverse of the homogeneous transformation matrix.
        The inverse of a homogeneous matrix is:
        [ R^T  -R^T * p ]
        [ 0    1        ]
        """
        R_inv = self.matrix[:3, :3].T  # Rotation part transpose
        p_inv = -np.dot(R_inv, self.matrix[:3, 3])  # Inverse of the translation part
        inv_matrix = np.eye(4)
        inv_matrix[:3, :3] = R_inv
        inv_matrix[:3, 3] = p_inv
        return inv_matrix

    def __mul__(self, other):
        """
        Overload the multiplication operator to handle:
        - HomogeneousTransformationMatrix * HomogeneousTransformationMatrix
        - HomogeneousTransformationMatrix * point (3D vector)
        """
        if isinstance(other, HomogeneousTransformationMatrix):
            # Matrix multiplication of two homogeneous transformation matrices
            result_matrix = np.dot(self.matrix, other.matrix)
            return HomogeneousTransformationMatrix(result_matrix[:3, :3], result_matrix[:3, 3])
        elif isinstance(other, np.ndarray) and other.shape == (3,):
            # Transform the 3D point (apply transformation to vector)
            point_homogeneous = np.append(other, 1)  # Convert point to homogeneous coordinates
            transformed_point = np.dot(self.matrix, point_homogeneous)
            return transformed_point[:3]  # Return only the 3D part
        else:
            raise ValueError(f"Invalid multiplication with type {type(other)}")
        
    def __eq__(self, other):
        """
        Check if two homogeneous transformation matrices are equal.
        This checks if both the rotation part (R) and the translation part (p) are the same.
        """
        if isinstance(other, HomogeneousTransformationMatrix):
            return np.allclose(self.matrix, other.matrix)
        return False

    def __repr__(self):
        """
        String representation of the homogeneous transformation matrix.
        """
        return f"HomogeneousTransformationMatrix(\n{self.matrix})"

    def from_twist(self):
        pass

    def to_twist(self):
        """
        Converts a twist (6D vector) to a 4x4 homogeneous transformation matrix using the exponential map.
        """
        pass
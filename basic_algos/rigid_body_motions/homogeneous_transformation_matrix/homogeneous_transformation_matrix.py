import numpy as np

from scipy.linalg import expm, logm
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

        return HomogeneousTransformationMatrix(R=RotationMatrix(R_inv), p=p_inv)
    
    def adjoint(self):
        """
        Computes the 6 by 6 adjoint representation [AdT] of the homogeneous transformation matrix T.
        The adjoint matrix is used to transform twists and wrenches between frames.
        """
        R = self.matrix[:3, :3]  # Rotation matrix
        p = self.matrix[:3, 3]   # Translation vector
        # Compute p_hat, the skew-symmetric matrix of p
        p_hat = np.array([
            [0, -p[2], p[1]],
            [p[2], 0, -p[0]],
            [-p[1], p[0], 0]
        ])
        # Build the 6x6 adjoint matrix
        adjoint_matrix = np.zeros((6, 6))
        adjoint_matrix[:3, :3] = R
        adjoint_matrix[3:, 3:] = R
        adjoint_matrix[3:, :3] = p_hat @ R  # p_hat * R for the translational part

        return adjoint_matrix

    def __mul__(self, other):
        """
        Overload the multiplication operator to handle:
        - HomogeneousTransformationMatrix * HomogeneousTransformationMatrix
        - HomogeneousTransformationMatrix * point (3D vector)
        """
        if isinstance(other, HomogeneousTransformationMatrix):
            # Matrix multiplication of two homogeneous transformation matrices
            result_matrix = np.dot(self.matrix, other.matrix)
            return HomogeneousTransformationMatrix(RotationMatrix(result_matrix[:3, :3]), result_matrix[:3, 3])
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
        return f"{self.matrix}"

    def from_se3mat(self, se3mat):
        """
        Convert from a se(3) matrix to a homogeneous transformation matrix (SE(3)).
        se3mat: 4x4 se(3) matrix (with skew-symmetric omega_hat and v)
        """
        assert se3mat.shape == (4, 4), "se(3) matrix must be 4x4."
        self.matrix = expm(se3mat)  # Exponential map to get SE(3)
        return self

    def from_twist(self, twist):
        """
        Convert from a 6D twist vector to a SE(3) matrix.
        twist: [omega_x, omega_y, omega_z, v_x, v_y, v_z]
        """
        omega = twist[:3]  # Angular velocity
        v = twist[3:]      # Linear velocity
        
        # Skew-symmetric matrix for angular velocity (omega_hat)
        omega_hat = np.array([
            [0, -omega[2], omega[1]],
            [omega[2], 0, -omega[0]],
            [-omega[1], omega[0], 0]
        ])
        
        # Construct the se(3) matrix
        se3_matrix = np.zeros((4, 4))
        se3_matrix[:3, :3] = omega_hat
        se3_matrix[:3, 3] = v
        
        # Convert to SE(3) using from_se3mat
        return self.from_se3mat(se3_matrix)
    
    def to_se3mat(self):
        """
        Convert the homogeneous transformation matrix (SE(3)) to a se(3) matrix.
        Returns a 4x4 se(3) matrix.
        """
        return logm(self.matrix)  # Logarithmic map to get se(3)
    
    def to_twist(self):
        """
        Convert the SE(3) matrix back to a 6D twist vector.
        Returns: [omega_x, omega_y, omega_z, v_x, v_y, v_z]
        """
        se3_matrix = self.to_se3mat()  # Convert SE(3) to se(3)
        
        # Extract angular velocity (omega_hat) and linear velocity (v)
        omega_hat = se3_matrix[:3, :3]
        v = se3_matrix[:3, 3]
        
        # Extract angular velocity from omega_hat
        omega = np.array([omega_hat[2, 1], omega_hat[0, 2], omega_hat[1, 0]])
        
        # Return combined twist vector [omega_x, omega_y, omega_z, v_x, v_y, v_z]
        return np.hstack([omega, v])
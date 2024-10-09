import numpy as np

class RotationMatrix:
    def __init__(self, matrix=None):
        """
        Initialize the rotation matrix. Default is the identity matrix.
        """
        if matrix is None:
            self.matrix = np.eye(3)  # Default to identity matrix
        else:
            self.matrix = np.array(matrix)
            assert self._is_valid_rotation_matrix(), "Not a valid rotation matrix."

    def _is_valid_rotation_matrix(self):
        """
        Check if the matrix is a valid rotation matrix.
        A valid rotation matrix is orthogonal and has a determinant of +1.
        """
        Rt = self.matrix.T
        shouldBeIdentity = np.dot(Rt, self.matrix)
        I = np.eye(3)
        return np.allclose(shouldBeIdentity, I) and np.isclose(np.linalg.det(self.matrix), 1.0)

    def transpose(self):
        """
        Return the transpose of the rotation matrix.
        Since rotation matrices are orthogonal, the transpose is also its inverse.
        """
        return RotationMatrix(self.matrix.T)

    def inverse(self):
        """
        Return the inverse of the rotation matrix (should be the same as the transpose).
        For rotation matrices, the inverse is equal to the transpose.
        """
        return self.transpose()

    def from_axis_angle(self, axis, angle):
        """
        Generate a rotation matrix from an axis and an angle.
        """
        pass

    def from_quaternion(self, quaternion):
        """
        Generate a rotation matrix from a quaternion.
        """
        pass

    def from_euler(self, roll, pitch, yaw):
        """
        Generate a rotation matrix from roll, pitch, and yaw (Euler angles).
        """
        pass

    def to_axis_angle(self):
        """
        Convert the rotation matrix to an axis and an angle.
        """
        pass

    def to_quaternion(self):
        """
        Convert the rotation matrix to a quaternion.
        """
        pass

    def to_euler(self):
        """
        Convert the rotation matrix to Euler angles.
        """
        pass

    def __mul__(self, other):
        """
        Overload the multiplication operator for:
        - RotationMatrix * RotationMatrix (composition of rotations)
        - RotationMatrix * vector (rotating a vector)
        
        By default, this performs a left multiplication.
        """
        if isinstance(other, RotationMatrix):
            # Left multiply the two rotation matrices
            return RotationMatrix(np.dot(self.matrix, other.matrix))
        elif isinstance(other, np.ndarray) and other.shape == (3,):
            # Left multiply the matrix by the vector (apply rotation to vector)
            return np.dot(self.matrix, other)
        else:
            raise ValueError(f"Invalid multiplication with type {type(other)}")

    def __eq__(self, other):
        """
        Check if two rotation matrices are equal.
        """
        return np.allclose(self.matrix, other.matrix)

    def __repr__(self):
        """
        String representation of the rotation matrix.
        """
        return f"{self.matrix}"

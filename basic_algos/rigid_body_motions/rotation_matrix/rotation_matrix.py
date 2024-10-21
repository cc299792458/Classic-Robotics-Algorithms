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
        Generate a rotation matrix from an axis and an angle using Rodrigues' Rotation Formula.

        Rodrigues' Rotation Formula:
        R = I + sin(theta) * [w]_x + (1 - cos(theta)) * [w]_x^2

        Where:
        - R is the resulting rotation matrix.
        - I is the identity matrix.
        - theta is the rotation angle (in radians).
        - w is the unit vector representing the axis of rotation.
        - [w]_x is the skew-symmetric matrix (hat operator) of the axis w.
        """
        # Normalize the axis to ensure it's a unit vector
        axis = np.array(axis)
        axis = axis / np.linalg.norm(axis)  # Ensure it's a unit vector

        # Compute terms for Rodrigues' formula
        I = np.eye(3)  # Identity matrix
        axis_hat = np.array([[0, -axis[2], axis[1]],
                            [axis[2], 0, -axis[0]],
                            [-axis[1], axis[0], 0]])  # Skew-symmetric matrix

        # Rodrigues' rotation formula
        self.matrix = I + np.sin(angle) * axis_hat + (1 - np.cos(angle)) * np.dot(axis_hat, axis_hat)

    def from_euler(self, roll, pitch, yaw):
        """
        Generate a rotation matrix from roll, pitch, and yaw (Euler angles).
        Generate a rotation matrix from roll, pitch, and yaw (Euler angles).
        The angles are assumed to be in ZYX order: yaw (Z), pitch (Y), roll (X).
        """
        # Compute rotation matrices around each axis
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw),  np.cos(yaw), 0],
                       [0,            0,           1]])

        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0,             1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])

        Rx = np.array([[1, 0,             0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll),  np.cos(roll)]])
        
        self.matrix = np.dot(np.dot(Rz, Ry), Rx)

    def from_quaternion(self, quaternion):
        """
        Generate a rotation matrix from a quaternion.
        """
        pass

    def to_axis_angle(self):
        """
        Convert the rotation matrix to an axis and an angle.

        The axis-angle representation of a rotation matrix R can be extracted as:
        - The rotation axis is the unit vector corresponding to the eigenvector of R associated with the eigenvalue 1.
        - The rotation angle is derived from the trace of the rotation matrix: 
        theta = arccos((trace(R) - 1) / 2)

        Returns:
        - axis: a 3D unit vector representing the rotation axis.
        - angle: the rotation angle (in radians).
        """
        # Compute the angle using the trace of the matrix
        angle = np.arccos((np.trace(self.matrix) - 1) / 2.0)

        # If the angle is small (close to zero), return a default axis
        if np.isclose(angle, 0):
            return np.array([1.0, 0.0, 0.0]), 0.0

        # If the angle is close to pi, special care is needed
        if np.isclose(angle, np.pi):
            # The rotation axis can be found from the diagonal elements
            # (R[i,i] = cos(theta) + (1 - cos(theta)) * n_i^2, where n is the axis)
            axis = np.sqrt((np.diag(self.matrix) + 1) / 2.0)
            return axis, np.pi

        # General case: calculate the rotation axis
        axis = np.array([self.matrix[2, 1] - self.matrix[1, 2],
                        self.matrix[0, 2] - self.matrix[2, 0],
                        self.matrix[1, 0] - self.matrix[0, 1]]) / (2 * np.sin(angle))

        return axis, angle

    def to_euler(self):
        """
        Convert the rotation matrix to Euler angles (ZYX order).
        Returns:
        - roll: rotation around X-axis
        - pitch: rotation around Y-axis
        - yaw: rotation around Z-axis
        """
        # Check for gimbal lock (singularity when pitch is ±90 degrees)
        if np.isclose(self.matrix[2, 0], -1.0):
            pitch = np.pi / 2
            yaw = 0
            roll = np.arctan2(self.matrix[0, 1], self.matrix[0, 2])
        elif np.isclose(self.matrix[2, 0], 1.0):
            pitch = -np.pi / 2
            yaw = 0
            roll = -np.arctan2(self.matrix[0, 1], self.matrix[0, 2])
        else:
            pitch = -np.arcsin(self.matrix[2, 0])
            roll = np.arctan2(self.matrix[2, 1] / np.cos(pitch), self.matrix[2, 2] / np.cos(pitch))
            yaw = np.arctan2(self.matrix[1, 0] / np.cos(pitch), self.matrix[0, 0] / np.cos(pitch))

        return roll, pitch, yaw

    def to_quaternion(self):
        """
        Convert the rotation matrix to a quaternion.
        """
        pass
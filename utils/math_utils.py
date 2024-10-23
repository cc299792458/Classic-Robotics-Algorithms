import numpy as np

def skew_symmetric(vector):
    """
    Converts a 3D vector into a skew-symmetric matrix.
    """
    return np.array([[0, -vector[2], vector[1]],
                     [vector[2], 0, -vector[0]],
                     [-vector[1], vector[0], 0]])

def line_intersects_rect(p1, p2, rect):
    """Check if a line segment from p1 to p2 intersects a rectangle defined by rect."""
    (rx1, ry1), (rx2, ry2) = rect  # Rectangle coordinates
    # Ensure rectangle coordinates are ordered
    rx_min, rx_max = sorted([rx1, rx2])
    ry_min, ry_max = sorted([ry1, ry2])
    
    # Check if both points are on one side of the rectangle
    if (p1[0] < rx_min and p2[0] < rx_min) or (p1[0] > rx_max and p2[0] > rx_max) or \
       (p1[1] < ry_min and p2[1] < ry_min) or (p1[1] > ry_max and p2[1] > ry_max):
        return False  # The line segment is completely outside the rectangle

    # Check if the line intersects any of the rectangle's sides
    rect_lines = [
        ((rx_min, ry_min), (rx_max, ry_min)),  # Bottom edge
        ((rx_max, ry_min), (rx_max, ry_max)),  # Right edge
        ((rx_max, ry_max), (rx_min, ry_max)),  # Top edge
        ((rx_min, ry_max), (rx_min, ry_min))   # Left edge
    ]

    for edge in rect_lines:
        if line_segments_intersect(p1, p2, edge[0], edge[1]):
            return True  # The line intersects an edge of the rectangle

    # Check if the line segment is completely inside the rectangle
    if (rx_min <= p1[0] <= rx_max and ry_min <= p1[1] <= ry_max) and \
       (rx_min <= p2[0] <= rx_max and ry_min <= p2[1] <= ry_max):
        return True  # The line segment is completely inside the rectangle

    return False  # No intersection

def line_segments_intersect(p1, p2, q1, q2):
    """Check if line segment p1-p2 intersects with line segment q1-q2."""
    def ccw(a, b, c):
        """Check if three points are listed in a counterclockwise order."""
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    return (ccw(p1, q1, q2) != ccw(p2, q1, q2)) and (ccw(p1, p2, q1) != ccw(p1, p2, q2))
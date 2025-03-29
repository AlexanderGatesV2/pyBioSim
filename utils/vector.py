import math

def normalize(vector):
    """
    Normalize a vector to unit length.
    
    Args:
        vector: A 2D vector as a tuple (x, y)
        
    Returns:
        Tuple: Normalized vector with length 1, or (0, 0) if input has zero length
    """
    magnitude = length(vector)
    if magnitude > 0:
        return (vector[0] / magnitude, vector[1] / magnitude)
    return (0, 0)

def length(vector):
    """
    Calculate the length (magnitude) of a vector.
    
    Args:
        vector: A 2D vector as a tuple (x, y)
        
    Returns:
        Float: The length of the vector
    """
    return math.sqrt(vector[0] ** 2 + vector[1] ** 2)

def dot_product(v1, v2):
    """
    Calculate the dot product of two vectors.
    
    Args:
        v1: First 2D vector as a tuple (x, y)
        v2: Second 2D vector as a tuple (x, y)
        
    Returns:
        Float: The dot product of the two vectors
    """
    return v1[0] * v2[0] + v1[1] * v2[1]


def angle_between(v1, v2):
    """
    Calculate the angle between two vectors in radians.
    
    Args:
        v1: First 2D vector as a tuple (x, y)
        v2: Second 2D vector as a tuple (x, y)
        
    Returns:
        Float: The angle between the vectors in radians (0 to π)
    """
    dot = dot_product(v1, v2)
    len1 = length(v1)
    len2 = length(v2)

    # Avoid division by zero
    if len1 == 0 or len2 == 0:
        return 0

    # Ensure the value is within the valid range for arccos
    cosine = max(-1.0, min(1.0, dot / (len1 * len2)))
    return math.acos(cosine)


def rotate(vector, angle):
    """
    Rotate a vector by the specified angle.
    
    Args:
        vector: A 2D vector as a tuple (x, y)
        angle: Rotation angle in radians (positive is counterclockwise)
        
    Returns:
        Tuple: The rotated vector
    """
    x = vector[0] * math.cos(angle) - vector[1] * math.sin(angle)
    y = vector[0] * math.sin(angle) + vector[1] * math.cos(angle)
    return (x, y)


def perpendicular(vector):
    """
    Return a vector perpendicular to the input vector.
    
    Returns the 90-degree counterclockwise rotation of the input vector.
    
    Args:
        vector: A 2D vector as a tuple (x, y)
        
    Returns:
        Tuple: A perpendicular vector with the same magnitude
    """
    return (-vector[1], vector[0])

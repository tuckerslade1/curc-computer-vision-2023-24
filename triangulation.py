import numpy as np
from scipy.optimize import minimize

#TODO: grab position and angle data from camera output, use gyroscope data to determine azimuth and elevation angles, error modeling to account for noise in real world data, constraint enforcement if objects are a known distance apart

def spherical_to_cartesian_multilaterate_triangulation(r, azimuth, elevation):
    """
    Convert spherical coordinates to cartesian.
    r: Radius or distance to the point
    azimuth: Angle on the XY plane from the X-axis (0 to 360 degrees)
    elevation: Angle from the XY plane to the point (0 to 180 degrees)
    """
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return np.array([x, y, z])

def triangulate_camera_position(distance_A, azimuth_A, elevation_A, distance_B, azimuth_B, elevation_B):
    azimuth_A = np.radians(azimuth_A)
    elevation_A = np.radians(elevation_A)
    azimuth_B = np.radians(azimuth_B)
    elevation_B = np.radians(elevation_B)

    A = spherical_to_cartesian_multilaterate_triangulation(distance_A, azimuth_A, elevation_A)
    B = spherical_to_cartesian_multilaterate_triangulation(distance_B, azimuth_B, elevation_B)

    # function to minimize distance between estimated positions of objects A and B from the perspective of two hypothetical camera locations
    def error_function(camera_position):
        camera_A_vector = A - camera_position
        camera_B_vector = B - camera_position
        error_A = np.linalg.norm(camera_A_vector) - distance_A
        error_B = np.linalg.norm(camera_B_vector) - distance_B
        return error_A**2 + error_B**2

    # for now we are using the midpoint of A and B for the initial guess of the camera position (more accuracy can be achieved by using a better initial guess)
    initial_guess = (A + B) / 2

    # use scipy's minimize function to find the best camera position
    result = minimize(error_function, initial_guess, method='BFGS')
    
    if result.success:
        return result.x  # The optimal camera position
    else:
        raise ValueError("Optimization failed: " + result.message)

# Example usage:
distance_A = 1000  # Distance to object A
azimuth_A = 45    # Azimuth angle to object A (in degrees)
elevation_A = 30  # Elevation angle to object A (in degrees)

distance_B = 1500  # Distance to object B
azimuth_B = 135   # Azimuth angle to object B (in degrees)
elevation_B = 10  # Elevation angle to object B (in degrees)

camera_position = triangulate_camera_position(
    distance_A, azimuth_A, elevation_A,
    distance_B, azimuth_B, elevation_B
)

print("Estimated Camera Position (Triangulation):", camera_position)


def spherical_to_cartesian_multilaterate(r, azimuth, elevation):
    """
    Convert spherical coordinates to cartesian coordinates.
    """
    azimuth = np.radians(azimuth)
    elevation = np.radians(elevation)
    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return np.array([x, y, z])

def multilaterate_with_angles(angles_distances, initial_guess=None):
    """
    Find the position of a point given angles (azimuth, elevation) and distances to known points.
    
    Args:
    - angles_distances: A list of tuples containing azimuth, elevation, and distance for each known point.
    - initial_guess: An initial guess for the camera position.
    
    Returns:
    - The estimated position of the camera.
    """
    if initial_guess is None:
        # use the centroid of the unit vectors pointing towards each object as initial guess
        initial_guess = np.mean([spherical_to_cartesian_multilaterate(1, *ad[:2]) for ad in angles_distances], axis=0)
    
    def objective_function(camera_position):
        total_error = 0
        for azimuth, elevation, distance in angles_distances:
            object_vector = spherical_to_cartesian_multilaterate(distance, azimuth, elevation)
            predicted_vector = object_vector - camera_position
            total_error += np.linalg.norm(predicted_vector)**2
        return total_error

    result = minimize(objective_function, initial_guess, method='L-BFGS-B')
    if result.success:
        return result.x
    else:
        raise ValueError("Optimization failed: " + result.message)

# Example usage with 4 known points:
angles_distances = [
    (45, 30, 500),    # Azimuth, Elevation, Distance to Object A
    (135, 45, 800),   # Azimuth, Elevation, Distance to Object B
    (225, 60, 1200),  # Azimuth, Elevation, Distance to Object C
    (315, 70, 300)    # Azimuth, Elevation, Distance to Object D
]

camera_position = multilaterate_with_angles(angles_distances)
print("Estimated Camera Position (Multilateration):", camera_position)
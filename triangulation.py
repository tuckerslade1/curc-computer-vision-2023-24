import numpy as np
from scipy.optimize import minimize

def spherical_to_cartesian(r, azimuth, elevation):
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

    A = spherical_to_cartesian(distance_A, azimuth_A, elevation_A)
    B = spherical_to_cartesian(distance_B, azimuth_B, elevation_B)

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

print("Estimated Camera Position:", camera_position)

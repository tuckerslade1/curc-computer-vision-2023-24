# calculate_object_measurements.py
import math
import numpy as np

# FOV of Intel RealSense d435i RGB camera is 69 x 42 deg
fovx = 69
fovy = 42

# Our current output resolution is 640x480, though this may change in the future
resx = 640
resy = 480

screen_centerx = resx / 2
screen_centery = resy / 2


# width, height, and angle are calculated using the formula d = 2*D*tan(delta/2)
# where d is measurement of object, D is distance to object, and delta is angular diameter in radians

# Calculate width of objects based on distance and angular diameter
def calculateWidth(depth_mm, x, x2):

    # angular diameter in x direction (deg)
    angx = fovx * (x2 - x) / resx

    # width (mm)
    return math.floor(2 * depth_mm * math.tan(math.radians(angx)/2))


def calculateHeight(depth_mm, y, y2):

    # angular diameter in y direction (deg)
    angy = fovy * (y2 - y) / resy

    # height
    return math.floor(2 * depth_mm * math.tan(math.radians(angy)/2))


def calculateAngle(cx, cy):

    # calculate angle
    object_angle = math.degrees(np.arctan2(resy - cy - screen_centery, cx - screen_centerx))

    # round
    object_angle *= 10
    object_angle = math.floor(object_angle)
    object_angle /= 10

    # make negative angles positive
    if object_angle < 0:
        object_angle += 360

    return object_angle
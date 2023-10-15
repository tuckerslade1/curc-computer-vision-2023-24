import cv2
import numpy as np
import math
from mask_rcnn import *

class InfoWindow:
    def __init__(self, position, title, measurements, color):
        self.position = position
        self.title = title.capitalize()
        self.measurements = measurements
        self.color = color

        # default width and height values for now (in px)
        self.width = 118
        self.height = 87
        
    # display transparent InfoWindows
    def display(self, frame):
        
        # blank mask
        shapes = np.zeros_like(frame, np.uint8)

        # window background
        cv2.rectangle(frame, self.position, (self.position[0] + self.width, self.position[1] + self.height), self.color, -1)

        # title
        cv2.putText(frame, self.title, (self.position[0] + 5, self.position[1] + 18), 0, 0.5, (255, 255, 255), 2)
        
        # distance from camera (cm)
        cv2.putText(frame, "d: {} cm".format(self.measurements[0]), (self.position[0] + 5, self.position[1] + 33), 0, 0.5, (255, 255, 255), 1)
        
        # width (cm)
        cv2.putText(frame, "w: {} cm".format(self.measurements[1]), (self.position[0] + 5, self.position[1] + 48), 0, 0.5, (255, 255, 255), 1)
        
        # height (cm)
        cv2.putText(frame, "h: {} cm".format(self.measurements[2]), (self.position[0] + 5, self.position[1] + 63), 0, 0.5, (255, 255, 255), 1)

        # angle (deg)
        cv2.putText(frame, "a: {} deg".format(self.measurements[3]), (self.position[0] + 5, self.position[1] + 78), 0, 0.5, (255, 255, 255), 1)
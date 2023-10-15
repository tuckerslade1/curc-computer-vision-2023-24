import cv2
import numpy as np
import math
from mask_rcnn import *

class InfoWindow:
    def __init__(self, frame, position, title, measurements, color, alpha):
        self.frame = frame
        self.position = position
        self.title = title.capitalize()
        self.measurements = measurements
        self.color = color
        self.alpha = alpha

        # default width and height values for now (in px)
        self.width = 118
        self.height = 87
        
    # draw method for standard object InfoWindows
    def display(self):
        
        # window background
        cv2.rectangle(self.frame, self.position, (self.position[0] + self.width, self.position[1] + self.height), self.color, -1)

        # title
        cv2.putText(self.frame, self.title, (self.position[0] + 5, self.position[1] + 18), 0, 0.5, (255, 255, 255), 2)
        
        # distance from camera (cm)
        cv2.putText(self.frame, "d: {} cm".format(self.measurements[0]), (self.position[0] + 5, self.position[1] + 33), 0, 0.5, (255, 255, 255), 1)
        
        # width (cm)
        cv2.putText(self.frame, "w: {} cm".format(self.measurements[1]), (self.position[0] + 5, self.position[1] + 48), 0, 0.5, (255, 255, 255), 1)
        
        # height (cm)
        cv2.putText(self.frame, "h: {} cm".format(self.measurements[2]), (self.position[0] + 5, self.position[1] + 63), 0, 0.5, (255, 255, 255), 1)

        # angle (deg)
        cv2.putText(self.frame, "a: {} deg".format(self.measurements[3]), (self.position[0] + 5, self.position[1] + 78), 0, 0.5, (255, 255, 255), 1)
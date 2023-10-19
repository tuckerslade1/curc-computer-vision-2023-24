import cv2
import numpy as np

window_width = 400

window_height = 200

grid_size = 20

minimap_frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)

def drawMinimapBackground(minimap_frame):

    #--grid--

    # columns
    for x in range(0, window_width, grid_size):
        cv2.line(minimap_frame, (x, 0), (x, window_height), (255,255,255), 1)
    
    # rows
    for y in range(0, window_height, grid_size):
        cv2.line(minimap_frame, (0, y), (window_width, y), (255,255,255), 1)

    # origin
    cv2.circle(minimap_frame, (int(window_width/2), int(window_height/2)), 3, (0,0,255), -1)

    return minimap_frame
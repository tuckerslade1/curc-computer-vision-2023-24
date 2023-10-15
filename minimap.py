import cv2
import numpy as np

window_width = 400

window_height = 400

grid_size = 20

minimap_frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)


def showMinimap():

    # create window
    cv2.imshow("MINIMAP", minimap_frame)


    # create grid

    # columns
    for x in range(0, window_width, grid_size):
        cv2.line(minimap_frame, (x, 0), (x, window_height), (255,255,255), 1)
    
    # rows
    for y in range(0, window_height, grid_size):
        cv2.line(minimap_frame, (0, y), (window_width, y), (255,255,255), 1)

    # origin
    cv2.drawMarker(minimap_frame, (int(window_width/2), int(window_height/2)), (0,0,255), cv2.MARKER_CROSS, 15, 1)
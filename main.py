# main.py
import cv2
from realsense_camera import *
from mask_rcnn import *

# Initialize camera
rs = RealsenseCamera()
mrcnn = MaskRCNN()



while True:
    
    #--bgr_frame--
    # Get frame in real time from RealSense camera
    ret, bgr_frame, depth_frame = rs.get_frame_stream()

    # Get object mask
    boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)

    # Draw object mask
    bgr_frame = mrcnn.draw_object_mask(bgr_frame)
    mrcnn.draw_object_info(bgr_frame, depth_frame)

    # Show bgr_frame
    cv2.imshow("camera", bgr_frame)


    #--minimaps--
    flat_minimap_frame = mrcnn.drawFlatMinimap(depth_frame)
    cv2.imshow("x-y (flat view)", flat_minimap_frame)

    birdseye_minimap_frame = mrcnn.drawBirdseyeMinimap(depth_frame)
    cv2.imshow("x-z (birdseye view)", birdseye_minimap_frame)





    cv2.waitKey(1)
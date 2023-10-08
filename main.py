import cv2
from realsense_camera import *
from mask_rcnn import *

# Initialize camera
rs = RealsenseCamera()
mrcnn = MaskRCNN()

while True:
    
    # Get frame in real time from RealSense camera
    ret, bgr_frame, depth_frame = rs.get_frame_stream()

    # Get object mask
    boxes, classes, contours, centers = mrcnn.detect_objects_mask(bgr_frame)

    # Draw object mask
    bgr_frame = mrcnn.draw_object_mask(bgr_frame)
    mrcnn.draw_object_info(bgr_frame, depth_frame)

    cv2.imshow("BGR FRAME", bgr_frame)

    cv2.waitKey(1)
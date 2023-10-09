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

    # Draw transparent objects onto bgr_frame

    # Initialize blank mask image of same dimensions for drawing the objects
    shapes = np.zeros_like(bgr_frame, np.uint8)

    # Draw transparent shapes
    cv2.rectangle(shapes, (5, 5), (100, 75), (255, 255, 255), cv2.FILLED)

    # Generate output by blending image with shapes image, using the shapes
    # images also as mask to limit the blending to those parts
    out = bgr_frame.copy()
    alpha = 0.5
    mask = shapes.astype(bool)
    out[mask] = cv2.addWeighted(bgr_frame, alpha, shapes, 1 - alpha, 0)[mask]

    cv2.imshow("BGR FRAME", out)

    cv2.waitKey(1)
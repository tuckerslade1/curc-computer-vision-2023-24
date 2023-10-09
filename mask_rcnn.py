# https://pysource.com/instance-segmentation-mask-rcnn-with-python-and-opencv
import cv2
import numpy as np
import math

class MaskRCNN:
    def __init__(self):
        # Loading Mask RCNN
        self.net = cv2.dnn.readNetFromTensorflow("dnn/frozen_inference_graph_coco.pb",
                                            "dnn/mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

        # Generate random colors
        np.random.seed(2)
        self.colors = np.random.randint(0, 255, (90, 3))

        # Conf threshold
        self.detection_threshold = 0.7
        self.mask_threshold = 0.3

        self.classes = []
        with open("dnn/classes.txt", "r") as file_object:
            for class_name in file_object.readlines():
                class_name = class_name.strip()
                self.classes.append(class_name)

        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []
        self.obj_contours = []

        # Distances
        self.distances = []


    def detect_objects_mask(self, bgr_frame):
        blob = cv2.dnn.blobFromImage(bgr_frame, swapRB=True)
        self.net.setInput(blob)

        boxes, masks = self.net.forward(["detection_out_final", "detection_masks"])

        # Detect objects
        frame_height, frame_width, _ = bgr_frame.shape
        detection_count = boxes.shape[2]

        # Object Boxes
        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []
        self.obj_contours = []

        for i in range(detection_count):
            box = boxes[0, 0, i]
            class_id = box[1]
            score = box[2]
            color = self.colors[int(class_id)]
            if score < self.detection_threshold:
                continue

            # Get box Coordinates
            x = int(box[3] * frame_width)
            y = int(box[4] * frame_height)
            x2 = int(box[5] * frame_width)
            y2 = int(box[6] * frame_height)
            self.obj_boxes.append([x, y, x2, y2])

            cx = (x + x2) // 2
            cy = (y + y2) // 2
            self.obj_centers.append((cx, cy))

            # append class
            self.obj_classes.append(class_id)

            # Contours
            # Get mask coordinates
            # Get the mask
            mask = masks[i, int(class_id)]
            roi_height, roi_width = y2 - y, x2 - x
            mask = cv2.resize(mask, (roi_width, roi_height))
            _, mask = cv2.threshold(mask, self.mask_threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(np.array(mask, np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.obj_contours.append(contours)

        return self.obj_boxes, self.obj_classes, self.obj_contours, self.obj_centers

    def draw_object_mask(self, bgr_frame):
        # loop through the detection
        for box, class_id, contours in zip(self.obj_boxes, self.obj_classes, self.obj_contours):
            x, y, x2, y2 = box
            roi = bgr_frame[y: y2, x: x2]
            roi_height, roi_width, _ = roi.shape
            color = self.colors[int(class_id)]

            roi_copy = np.zeros_like(roi)

            for cnt in contours:
                # cv2.f(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
                cv2.drawContours(roi, [cnt], - 1, (int(color[0]), int(color[1]), int(color[2])), 3)
                cv2.fillPoly(roi_copy, [cnt], (int(color[0]), int(color[1]), int(color[2])))
                roi = cv2.addWeighted(roi, 1, roi_copy, 0.5, 0.0)
                bgr_frame[y: y2, x: x2] = roi
        return bgr_frame

    def draw_object_info(self, bgr_frame, depth_frame):
        # loop through the detection
        for box, class_id, obj_center in zip(self.obj_boxes, self.obj_classes, self.obj_centers):
            x, y, x2, y2 = box

            color = self.colors[int(class_id)]
            color = (int(color[0]), int(color[1]), int(color[2]))

            cx, cy = obj_center

            depth_mm = depth_frame[cy, cx]


            # Calculate width of objects based on distance and angular diameter

            fovx = 69 # Horizontal FOV in degrees of Intel RealSense d435i RGB camera
            fovy = 42 # Vertical FOV in degrees of Intel RealSense d435i RGB camera

            # Our current output resolution is 640x480, though this may change in the future
            resx = 640
            resy = 480

            # Calculate the angular diameter of objects in degrees
            angx = fovx * (x2 - x) / resx
            angy = fovy * (y2 - y) / resy

            # Calculate the actual width and height of objects in mm
            # Given by formula d = 2*D*tan(delta/2),
            # Where d is measurement of object, D is distance to object, and delta is angular diameter in radians
            width_mm = math.floor(2 * depth_mm * math.tan(math.radians(angx)/2))
            height_mm = math.floor(2 * depth_mm * math.tan(math.radians(angy)/2))

            # Round angles to one decimal point
            angx *= 10
            angx = math.floor(angx)
            angx /= 10

            angy *= 10
            angy = math.floor(angy)
            angy /= 10

            # Calculate angles from center of screen to center of objects
            screen_centerx = resx / 2 # x-pos of screen center
            screen_centery = resy / 2 # y-pos of screen center
            object_angle = math.degrees(np.arctan2(resy - cy - screen_centery, cx - screen_centerx))

            # Round angles to one decimal point
            object_angle *= 10
            object_angle = math.floor(object_angle)
            object_angle /= 10

            # Make output angles positive
            if object_angle < 0:
                object_angle += 360


            # DRAW

            # Draw background rectangles
            cv2.rectangle(bgr_frame, (x, y), (x + 118, y + 87), color, -1)

            # Draw outline of background rectangles
            #cv2.rectangle(bgr_frame, (x, y), (x2, y2), color, 1)

            # Draw lines through center of objects
            # cv2.line(bgr_frame, (cx, y), (cx, y2), color, 1)
            # cv2.line(bgr_frame, (x, cy), (x2, cy), color, 1)

            # Draw (x,y) coordinates of corners
            # cv2.putText(bgr_frame, str(x) + ", " + str(y), (x - 3, y - 3), 0, 0.5, (255, 255, 255), 1)
            # cv2.putText(bgr_frame, str(x2) + ", " + str(y), (x2 + 3, y - 3), 0, 0.5, (255, 255, 255), 1)
            # cv2.putText(bgr_frame, str(x) + ", " + str(y2), (x - 3, y2 + 15), 0, 0.5, (255, 255, 255), 1)
            # cv2.putText(bgr_frame, str(x2) + ", " + str(y2), (x2 + 3, y2 + 15), 0, 0.5, (255, 255, 255), 1)

            # Draw object type and measurements

            class_name = self.classes[int(class_id)]
            cv2.putText(bgr_frame, class_name.capitalize(), (x + 5, y + 18), 0, 0.5, (255, 255, 255), 2)
            # d: represents measured distance to the center of object in cm
            cv2.putText(bgr_frame, "d: {} cm".format(depth_mm / 10), (x + 5, y + 33), 0, 0.5, (255, 255, 255), 1) # distance
            # w: represents calculated width of object in cm
            cv2.putText(bgr_frame, "w: {} cm".format(width_mm / 10), (x + 5, y + 48), 0, 0.5, (255, 255, 255), 1) # width
            # h: represents calculated height of object in cm
            cv2.putText(bgr_frame, "h: {} cm".format(height_mm / 10), (x + 5, y + 63), 0, 0.5, (255, 255, 255), 1) # height
            # ang: represents angle from center of screen to center of object in degrees
            cv2.putText(bgr_frame, "ang: {} deg".format(object_angle), (x + 5, y + 78), 0, 0.5, (255, 255, 255), 1) # height

            # Draw marker at center of screen
            cv2.drawMarker(bgr_frame, (int(screen_centerx), int(screen_centery)), (0,0,255), cv2.MARKER_CROSS, 999, 1)

            # Draw lines from center of screen to objects to confirm ang values
            cv2.arrowedLine(bgr_frame, (int(screen_centerx), int(screen_centery)), (cx, cy), (color[0]/2,color[1]/2,color[2]/2), 1, cv2.LINE_AA)



        return bgr_frame






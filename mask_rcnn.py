# mask_rcnn.py
# https://pysource.com/instance-segmentation-mask-rcnn-with-python-and-opencv
import cv2
import numpy as np
from info_frames import *
from calculate_object_measurements import *

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
                contour_alpha = 0.2
                # cv2.f(roi, [cnt], (int(color[0]), int(color[1]), int(color[2])))
                cv2.drawContours(roi, [cnt], - 1, (int(color[0]), int(color[1]), int(color[2])), 3)
                cv2.fillPoly(roi_copy, [cnt], (int(color[0]), int(color[1]), int(color[2])))
                roi = cv2.addWeighted(roi, 1, roi_copy, contour_alpha, 0.0)
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


            # --CALCULATE--
            width_mm = calculateWidth(depth_mm, x, x2)
            
            height_mm = calculateHeight(depth_mm, y, y2)
            
            object_angle = calculateAngle(cx, cy)


            # --DRAW--
            class_name = self.classes[int(class_id)]

            # mask
            shapes = np.zeros_like(bgr_frame, np.uint8)

            # display InfoWindows on mask
            current_window = InfoWindow([x, y], class_name, [depth_mm/10, height_mm/10, width_mm/10, object_angle], color)
            current_window.display(bgr_frame)

            # crosshair at screen center
            cv2.drawMarker(shapes, (int(screen_centerx), int(screen_centery)), (50, 50, 50), cv2.MARKER_CROSS, 30, 2)

            # vectors to object centers
            cv2.arrowedLine(shapes, (int(screen_centerx), int(screen_centery)), (cx, cy), (color[0],color[1],color[2]), 1, cv2.LINE_AA)

            # blend mask with bgr_frame
            alpha = 1
            mask = shapes.astype(bool)
            bgr_frame[mask] = cv2.addWeighted(bgr_frame, 1, shapes, alpha, 0)[mask]

        return bgr_frame

    def drawFlatMinimap(self, depth_frame):

        window_width = 400
        window_height = 200
        grid_size = 20

        # draw background
        minimap_frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)

        # columns
        for x in range(0, window_width, grid_size):
            cv2.line(minimap_frame, (x, 0), (x, window_height), (255,255,255), 1)
        
        # rows
        for y in range(0, window_height, grid_size):
            cv2.line(minimap_frame, (0, y), (window_width, y), (255,255,255), 1)

        # origin
        cv2.circle(minimap_frame, (int(window_width/2), int(window_height/2)), 3, (0,0,255), -1)

        # loop through the detection
        for box, class_id, obj_center in zip(self.obj_boxes, self.obj_classes, self.obj_centers):
            x, y, x2, y2 = box

            color = self.colors[int(class_id)]
            color = (int(color[0]), int(color[1]), int(color[2]))

            cx, cy = obj_center

            # add objects to minimap
            minimap_object_location = (int((cx/640) * window_width), int((cy/480)*window_height))
            cv2.circle(minimap_frame, minimap_object_location, 5, color, -1)

        return minimap_frame
    
    def drawBirdseyeMinimap(self, depth_frame):
        
        window_width = 400
        window_height = 200
        grid_size = 20
        max_depth = 200 # cm

        # draw background
        minimap_frame = np.zeros((window_height, window_width, 3), dtype=np.uint8)

        # columns
        for x in range(0, window_width, grid_size):
            cv2.line(minimap_frame, (x, 0), (x, window_height), (255,255,255), 1)
        
        # rows
        for y in range(0, window_height, grid_size):
            cv2.line(minimap_frame, (0, y), (window_width, y), (255,255,255), 1)

        # origin
        cv2.circle(minimap_frame, (int(window_width/2), window_height-3), 3, (0,0,255), -1)

        # loop through the detection
        for box, class_id, obj_center in zip(self.obj_boxes, self.obj_classes, self.obj_centers):
            x, y, x2, y2 = box

            color = self.colors[int(class_id)]
            color = (int(color[0]), int(color[1]), int(color[2]))

            cx, cy = obj_center

            depth_mm = depth_frame[cy, cx]

            # add objects to minimap
            minimap_object_location = (int((cx/640) * window_width), window_height - int((depth_mm/10) / (max_depth/window_height)))
            cv2.circle(minimap_frame, minimap_object_location, 5, color, -1)

        return minimap_frame
import cv2
import numpy as np
import keyboard
from ultralytics import YOLO
import time
from sort import Sort
from ultralytics import RTDETR
import torch
from transformers import DetrForObjectDetection, DetrImageProcessor
from queue import Queue
from threading import Thread
import multiprocessing
from transformers import RTDetrForObjectDetection, RTDetrImageProcessor

class Camera:
    def __init__(self, number, fps=10):
        self.number = number
        self.name = "Camera " + str(number)
        self.url = (rf"rtsp://admin:hik-2022@192.168.100.3:554/Streaming/Channels/{number}01")
        self.status = self.get_status()
        self.fps = fps
        self.model = None
        self.frame_queue = multiprocessing.Queue(maxsize=900)
        self.running = True
        self.capture_thread = None
    
    def start_thread(self):
        """
        Start the thread for capturing frames
        Returns:
        None 'Just starts the thread'
        """
        self.capture_thread = Thread(target=self.read_frames, daemon=True)
        self.capture_thread.start()

    def stop_thread(self):
        """
        Stop the thread for capturing frames
        Returns:
        None 'Just stops the thread'
        """
        self.running = False
        if self.capture_thread:
            self.capture_thread.join()

    def read_frames(self):
        """
        Read frames from the camera stream using a single thread
        Returns:
        None 'Just reads the frames'
        """
        cap = cv2.VideoCapture(self.url)

        while self.running:
            ret, frame = cap.read()
            if not ret:
                break
            if self.frame_queue.full():
                self.frame_queue.get()
            self.frame_queue.put(frame)

        cap.release()
    
    
    def get_status(self):
        """
        Get the status of the camera (Online/Offline)
        Returns:
        str: The status of the camera
        """
        cap = cv2.VideoCapture(self.url)
        if cap.isOpened():
            cap.release()
            return "Online"
        else:
            cap.release()
            return "Offline"
    
    def initialize_model(self):
        """
        Initialize the model for object detection
        Returns:
        model: The initialized model
        """
        if self.number == 1:
            model = YOLO(rf"/home/yousof/Desktop/runs/detect/Cam1/weights/best.pt")
            imgsz = 1024
            return model, imgsz
        elif self.number == 2:
            image_processor = RTDetrImageProcessor.from_pretrained("/home/yousof/Downloads/custom-model4-20250330T103957Z-001/custom-model4")
            model = RTDetrForObjectDetection.from_pretrained("/home/yousof/Downloads/custom-model4-20250330T103957Z-001/custom-model4")
            model.to("cuda")
            imgsz = 640
            classes = {0: "car", 1: "person"}
            return model, imgsz, image_processor, classes
        elif self.number == 3:
            model = YOLO(rf"/home/yousof/Desktop/runs/detect/Cam3/weights/best.pt")
            imgsz = 1024
            classes = {0: "car", 1: "person"}
            return model, imgsz
        else:
            model = YOLO("yolov8n.pt")
            imgsz = 640
            classes = {0: "car", 1: "person"}
            return model, imgsz
    
    
    def display(self):
        """
        Display the camera stream
        Returns:
        None 'Just displays the camera stream'
        """
        cap = cv2.VideoCapture(self.url)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Write at top right corner of the frame
            cv2.putText(frame, "Press R to start recording", (frame.shape[1] - 450, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
            # Display the frame
            cv2.namedWindow("Camera Stream", cv2.WINDOW_NORMAL)
            cv2.imshow('Camera Stream', frame)

            # Press 'q' to exit the loop
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # Press 'r' to start recording
            if cv2.waitKey(1) & 0xFF == ord('r'):
                cv2.destroyAllWindows()
                self.record(display=True)
                break
            
        cap.release()
        cv2.destroyAllWindows()
    
    def record(self, filename="output.mp4", display=False):
        """
        Record the camera stream can be called by display function by pressing 'r' and stopped by pressing 'q' which will save the recording as the filename provided.
        Args:
        filename (str): The name of the output file
        display (bool): Display the recording status
        
        Returns:
        None 'Just records the camera stream'
        """
        cap = cv2.VideoCapture(self.url)
        ret, frame = cap.read()
        output = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), 15, (frame.shape[1], frame.shape[0]))

        print("Recording... Press 'q' to stop recording")
        while True:
        
            if not ret:
                break
            output.write(frame)

            if display:
                cv2.circle(frame, (frame.shape[1] - 320, 30), 10, (0, 0, 255), -1)
                cv2.putText(frame, "Recording", (frame.shape[1] - 300, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.putText(frame, "Press 'q' to stop recording", (frame.shape[1] - 450, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                cv2.namedWindow("Camera Stream", cv2.WINDOW_NORMAL)
                cv2.imshow('Camera Stream', frame)
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Recording stopped by user.")
                    break
                    
            else:
                if keyboard.is_pressed('q'):  # Capturing the 'q' keypress
                    print("Recording stopped by user.")
                    break
                
            ret, frame = cap.read()

        cap.release()
        output.release()
        cv2.destroyAllWindows()

    def detect(self, write=False, filename="detection.mp4", threshold=0.5, display=True, iou=0.5, draw=True, track=True, Analyse=False, projection=False, flags={}, cubes=[], shared_data={}):
        """
        Detect objects in the camera stream

        Args:
        write (bool): Write the output to a file
        threshold (float): Confidence threshold
        display (bool): Display the output
        iou (float): Intersection over Union threshold
        draw (bool): Draw the bounding boxes on the frame
        track (bool): Track the objects in the frame
        Analyse (bool): Analyse the objects in the frame
        shared_data (dict): A dictionary to store shared data between processes
        projection (bool): Project the objects to the top view
        flags (dict): A dictionary of flags to control the projection
        cubes (list): A list of points to be projected 

        Returns:
        None
        """
        if self.number == 2: # RT-DETR
            model, imgsz, imageProcessor, classes = self.initialize_model()
        else:
            model, imgsz = self.initialize_model()
            classes = model.names

        self.start_thread() # Start the thread for capturing frames
        
        while True: # Wait for the first frame
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                break

        skip = False
        frame_time = 1 / 15
        First_Start = time.time()
        frame_counter = 0
        
        if track:
            tracker = Sort(max_age=25, min_hits=1, iou_threshold=0.2)  # Initialize the SORT tracker

        if write:
            output = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), 15, (frame.shape[1], frame.shape[0]))

        if Analyse and self.number == 4:
            box_area = self.InitializeInOutHouse()

        if projection:
            if self.number == 1:
                homography_matrix = np.array([[ 2.04672515e-02, -8.48785847e-03,  2.56547037e+02],
                                                [-4.05332877e-01,  4.28313480e-01,  7.25008701e+02],
                                                [-4.46144864e-04,  1.15867108e-03,  1.00000000e+00]])
            elif self.number == 2:
                homography_matrix = np.array([[-1.20471032e+00,  2.41704430e+00, -2.73333414e+01],
                                                [-5.77841615e-02,  2.92619286e+00, -9.48424194e+02],
                                                [-2.49322482e-03,  1.14407240e-02,  1.00000000e+00]])
            elif self.number == 3: 
                homography_matrix = np.array([[-2.25083434e-01,  2.00520109e-02,  3.27833955e+02],
                                                [-2.10716311e-01,  1.41898924e-01,  1.84141253e+02],
                                                [-5.80459009e-04,  3.61096970e-04,  1.00000000e+00]])
            else:
                print("Homography matrix not found for this camera, please change the projection parameter to False.")
                self.running = False

        print("Detecting... Press 'q' to stop detection")
        while self.running:
            if skip:
                print("Skipping...")
                skip = False
                frame_counter += 1
                # Remove the first frame from the queue
                if not self.frame_queue.empty():
                    self.frame_queue.get()
                continue

            while True:
                if not self.frame_queue.empty():
                    frame = self.frame_queue.get()
                    break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            if self.number == 2:
                results = self.process_rtDETR_frame(frame, model, imageProcessor)
            if self.number != 2:
                results = model.predict(frame, iou=iou, conf=threshold, verbose=False, imgsz=imgsz)
            
            for result in results:
                boxes = result.boxes
                detections = []
                for box in boxes:
                    if self.number == 2:
                        x1, y1, x2, y2 = box[:4]
                        conf = box[4]
                        cls = box[5] 
                    else:
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        cls = box.cls.item()
                        conf = box.conf.item()
                    detections.append([x1, y1, x2, y2, conf, cls, self.number])

                    if projection and not track:
                        self.projection_manager(homography_matrix, (x1, y1, x2, y2), flags, cubes, None, cls)

                if projection and not track:
                    flags[self.number].value = True # Reset the flag to stop adding new positions to the cubes list
                    # This flag will prevent the addition of new positions to the cubes list until they are visualized.

                detections = np.array(detections)

                if track:
                    TrackResults = self.tracking(detections, tracker, shared_data)

                    if projection:
                        for obj in TrackResults:
                            self.projection_manager(homography_matrix, (obj[0], obj[1], obj[2], obj[3]), flags, cubes, obj[4], obj[5])
                    flags[self.number].value = True # Reset the flag to stop adding new positions to the cubes list
                    # This flag will prevent the addition of new positions to the cubes list until they are visualized.

                    if draw and Analyse: # Draw and Analyse
                        if self.number == 2:
                            # Draw the line
                            cv2.line(frame, (1050, 380), (1450, 690), (0, 0, 255), 2)
                            self.Analyse(tracker, TrackResults, shared_data) # This will adjust the position of the object and will affect the color of the bounding box.
                            # Draw color based on the position of the object
                            for obj in TrackResults:
                                if self.is_car(obj): # If the object is a car
                                    id = obj[4] - 1 # Because the id is 1-based
                                    id = int(id)

                                    tracker_to_draw = next((t for t in tracker.trackers if t.id == id), None)

                                    if tracker_to_draw:

                                        if tracker_to_draw.position == "Right":
                                            color = (0, 0, 255)
                                        else:
                                            color = (255, 0, 0)

                                        frame = self.Drawing(frame, obj, classes, with_id=True, box_color=color, class_color=color, id_color=color)
                                        
                                else: # If the object is not a car
                                    frame = self.Drawing(frame, obj, classes, with_id=True)

                        elif self.number == 4:
                            self.Analyse(tracker, TrackResults, shared_data, box_area)

                            for obj in TrackResults:
                                if self.is_person(obj): # If the object is a person
                                    id = obj[4] - 1 # Because the id is 1-based
                                    id = int(id)

                                    tracker_to_draw = next((t for t in tracker.trackers if t.id == id), None)

                                    if tracker_to_draw:

                                        if tracker_to_draw.position == "Inside":
                                            color = (0, 0, 255)
                                        else:
                                            color = (255, 0, 0)

                                        frame = self.Drawing(frame, obj, classes, with_id=True, box_color=color, class_color=color, id_color=color)
                                else:
                                    frame = self.Drawing(frame, obj, classes, with_id=True)

                        elif self.number == 6:
                            self.Analyse(tracker, TrackResults, shared_data)
                            for obj in TrackResults:
                                frame = self.Drawing(frame, obj, classes, with_id=True)
                        
                        else:
                            for obj in TrackResults:
                                frame = self.Drawing(frame, obj, classes, with_id=True)

                    elif Analyse: # Analyse without drawing
                        if self.number == 2:
                            self.Analyse(tracker, TrackResults, shared_data)
                        elif self.number == 4:
                            self.Analyse(tracker, TrackResults, shared_data, box_area)
                        elif self.number == 6:
                            self.Analyse(tracker, TrackResults, shared_data)
                        else:
                            pass

                    else: # Draw without Analyse
                        for tracked in TrackResults:
                            frame = self.Drawing(frame, tracked, classes, with_id=True)
                        

                elif draw: # If detection only without tracking
                    for detection in detections:
                        frame = self.Drawing(frame, detection, classes, with_id=False)
                    
                if write:
                    output.write(frame)
                if display:
                    cv2.namedWindow("Camera Stream", cv2.WINDOW_NORMAL)
                    cv2.imshow("Camera Stream", frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            elapsed_time = time.time() - First_Start

            # Calculate the time at which the next frame should be displayed
            target_time = (frame_counter + 1) * frame_time
            threshold_time = (frame_counter + 90) * frame_time

            if elapsed_time > threshold_time:
                skip = True

            
            wait_time = target_time - elapsed_time
            # If wait_time is positive, sleep for that duration
            if wait_time > 0:
                time.sleep(wait_time)

            frame_counter += 1
        
        cv2.destroyAllWindows()

        
    
    def tracking(self, detections, tracker, shared_data={}):
        """
        Update the tracker with the detections and return the results
        Args:
        detections (numpy.ndarray): An array of detections of the form [x1, y1, x2, y2, confidence, object_class, Camera_number], [...], ...]
        tracker (sort.Sort): The SORT tracker
        shared_data (dict): A dictionary to store shared data between processes

        Returns:
        numpy.ndarray: An array of tracked objects of the form [[x1, y1, x2, y2, id, class_id, hit_streak, camera_id], [...], ...]
        """
        if detections.any() == True:
            TrackResults = tracker.update(detections[:, :7], shared_data)  # Pass the bounding box and confidence score
        else:
            TrackResults = tracker.update(np.empty((0, 7)), shared_data)
        
        return TrackResults
    
    def Analyse(self, tracker, TrackResults, shared_data, box_area=None):
        """
        Analyse the objects in the frame and update the shared data
        Args:
        tracker (sort.Sort): The SORT tracker
        TrackResults (numpy.ndarray): An array of tracked objects of the form [[x1, y1, x2, y2, id, class_id, hit_streak, camera_id], [...], ...]
        shared_data (dict): A dictionary to store shared data between processes
        box_area (numpy.ndarray): An array of points defining the house area in the form [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] 'used only for Camera 4'

        Returns:
        None 'Just updates the shared data'
        """
        shared_data["Right Now Outside"] = 0
        if TrackResults.size == 0:
            if self.number == 4:
                self.Analyse_Cam_4(None, [], shared_data, None, None)

            elif self.number == 6:
                self.Analyse_Cam_6(None, [], shared_data, None)

        for obj in TrackResults:
            if self.number == 2:
                self.Analyse_Cam_2(tracker, shared_data, obj)
            
            elif self.number == 4:
                self.Analyse_Cam_4(tracker, TrackResults, shared_data, obj, box_area)

            elif self.number == 6:
                self.Analyse_Cam_6(tracker, TrackResults, shared_data, obj)
                

    def Analyse_Cam_2(self, tracker, shared_data, obj):
        """
        Analyse the objects in the frame from Camera 2 and update the shared data
        Args:
        tracker (sort.Sort): The SORT tracker
        shared_data (dict): A dictionary to store shared data between processes
        obj (numpy.ndarray): An array of the form [x1, y1, x2, y2, id, class_id, hit_streak, camera_id] 'single element of TrackResults'

        Returns:
        None 'Just updates the shared data'
        """
        id = obj[4] - 1 # Because the id is 1-based
        id = int(id)
        position = self.check_car_position(obj[:4], obj[7])

        tracker_to_update = next((t for t in tracker.trackers if t.id == id), None)

        if tracker_to_update:
            if tracker_to_update.position is None:
                tracker_to_update.position = position

            elif tracker_to_update.position == "Right" and position != "Right":
                tracker_to_update.position = position
                shared_data["Right_to_left"] += 1

            elif tracker_to_update.position == "Left" and position != "Left":
                tracker_to_update.position = position
                shared_data["Left_to_right"] += 1
        else:
            print(f"No tracker found with id: {id}")
            for trk in tracker.trackers:
                print(trk.id)


    def Analyse_Cam_4(self, tracker, TrackResults, shared_data, obj, box_area):
        """
        Analyse the objects in the frame from Camera 4 and update the shared data
        Args:
        tracker (sort.Sort): The SORT tracker
        TrackResults (numpy.ndarray): An array of tracked objects of the form [[x1, y1, x2, y2, id, class_id, hit_streak, camera_id], [...], ...]
        shared_data (dict): A dictionary to store shared data between processes
        obj (numpy.ndarray): An array of the form [x1, y1, x2, y2, id, class_id, hit_streak, camera_id] 'single element of TrackResults'
        box_area (numpy.ndarray): An array of points defining the house area in the form [(x1, y1), (x2, y2), (x3, y3), (x4, y4)] 'used only for Camera 4'
        
        Returns:
        None 'Just updates the shared data'
        """
        found = False
        for i in TrackResults:
            if i[5] == 0:
                found = True
        if obj is not None:
            id = obj[4] - 1
            id = int(id)
            if self.is_person(obj):
                position = self.check_person_house_position(box_area, (obj[2], obj[3]))

                tracker_to_update = next((t for t in tracker.trackers if t.id == id), None)

                if tracker_to_update:
                    if tracker_to_update.position is None and position == "Inside":
                        tracker_to_update.position = position
                    
                    elif tracker_to_update.position is None and position == "Outside":
                        tracker_to_update.position = position

                    elif tracker_to_update.position == "Inside" and position != "Inside":
                        tracker_to_update.position = position
                        shared_data["Left The House"] += 1
                        shared_data["logs"].append(f"Person {id} left the house at {time.ctime()}")
                        shared_data["Right Now Inside"] -= 1
                    
                    elif tracker_to_update.position == "Outside" and position != "Outside":
                        tracker_to_update.position = position
                        shared_data["Entered The House"] += 1
                        shared_data["logs"].append(f"Person {id} entered the house at {time.ctime()}")
                        shared_data["Right Now Inside"] += 1
                    
                    if tracker_to_update.position == "Outside" and tracker_to_update.hit_streak >= 1:
                        shared_data["Right Now Outside"] += 1 # It will be initialized to 0 for each frame.
                        
            elif found:
                if not shared_data["Is car 1 at the house?"]: 
                    shared_data["logs"].append(f"Car 1 arrived at the house at {time.ctime()}")
                shared_data["Is car 1 at the house?"] = True
        else:
            if shared_data["Is car 1 at the house?"]:
                shared_data["logs"].append(f"Car 1 left the house at {time.ctime()}")
            shared_data["Is car 1 at the house?"] = False

    def Analyse_Cam_6(self, tracker, TrackResults, shared_data, obj):
        """
        Analyse the objects in the frame from Camera 6 and update the shared data
        Args:
        tracker (sort.Sort): The SORT tracker
        TrackResults (numpy.ndarray): An array of tracked objects of the form [[x1, y1, x2, y2, id, class_id, hit_streak, camera_id], [...], ...]
        shared_data (dict): A dictionary to store shared data between processes
        obj (numpy.ndarray): An array of the form [x1, y1, x2, y2, id, class_id, hit_streak, camera_id] 'single element of TrackResults'

        Returns:
        None 'Just updates the shared data'
        """
        found = False
        for i in TrackResults:
            if i[5] == 0:
                found = True
        if obj is not None:
            id = obj[4] - 1
            id = int(id)
            if self.is_person(obj):
                tracker_to_update = next((t for t in tracker.trackers if t.id == id), None)
                if tracker_to_update:
                    if tracker_to_update.position is None:
                        tracker_to_update.position = "Outside"
                        shared_data["Right Now Outside"] += 1
            
            elif found:
                if not shared_data["Is car 2 at the house?"]:
                    shared_data["logs"].append(f"Car 2 arrived at the house at {time.ctime()}")
                shared_data["Is car 2 at the house?"] = True
        else:
            if shared_data["Is car 2 at the house?"]:
                shared_data["logs"].append(f"Car 2 left the house at {time.ctime()}")
            shared_data["Is car 2 at the house?"] = False
                            
    def Drawing(self, frame, obj, classes, with_id, box_color=(0, 255, 0), class_color=(0, 255, 0), id_color=(0, 255, 0), font=cv2.FONT_HERSHEY_SIMPLEX, thickness=2):
        """
        Draw the bounding boxes on the frame
        Args:
        frame (numpy.ndarray): The frame to draw the bounding boxes on
        obj (numpy.ndarray): An array of the form [x1, y1, x2, y2, id, class_id, hit_streak, camera_id] 'single element of TrackResults'
        classes (list): A list of class names
        with_id (bool): Whether to display the id of the object
        box_color (tuple): The color of the bounding box
        class_color (tuple): The color of the class name
        id_color (tuple): The color of the id
        font (cv2.FONT): The font to use for the text
        thickness (int): The thickness of the bounding box and text

        Returns:
        numpy.ndarray: The frame with the bounding boxes drawn
        """
        x1, y1, x2, y2 = obj[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
        cv2.putText(frame, classes[int(obj[5])], (x1, y1 + 20), font, 0.5, class_color, thickness)
        if with_id:
            id = int(obj[4]) 
            cv2.putText(frame, f"ID: {id}", (x1, y1 - 10), font, 0.5, id_color, thickness)

        return frame
    
    def projection_manager(self, homography_matrix, xy, flags, cubes, Track_id=None, class_id=None):
        """
        Project the points to the top view
        Args:
        homography_matrix (numpy.ndarray): The homography matrix
        xy (tuple): The (x, y) coordinates of the point
        flags (dict): A dictionary of flags
        cubes (list): A list of points to be projected

        Returns:
        None 'Just updates the cubes list'
        """
        if not flags[self.number].value:
            if class_id == 0: # If the object is a car
                point = self.calculate_middle_point(xy)
            elif class_id == 1: # If the object is a person
                # Get the middle of the bottom of the bounding box
                x1, y1, x2, y2 = xy
                point = (int((x1 + x2) / 2), y2)
            
            original_point = np.array([point], dtype=np.float16)

            # Convert to homogeneous coordinates
            original_point_homogeneous = np.append(original_point, 1)  # [x, y, 1]

            # Apply the homography matrix
            projected_point_homogeneous = np.dot(homography_matrix, original_point_homogeneous)

            # Convert back to Cartesian coordinates
            projected_point = projected_point_homogeneous[:2] / projected_point_homogeneous[2]

            if not flags["Visualizer"].value:
                if self.number == 1:
                    if class_id == 0:
                        cubes.append([projected_point[0], projected_point[1], Track_id, "person"])
                else:
                    if class_id == 0:
                        cubes.append([projected_point[0], projected_point[1], Track_id, "car"])
                    else:
                        cubes.append([projected_point[0], projected_point[1], Track_id, "person"])
            else:
                pass

        return 

    
    def calculate_middle_point(self, coordinates):
        """
        Calculate the middle point of the bounding box
        Args:
        coordinates (tuple): The (x1, y1, x2, y2) coordinates of the bounding box

        Returns:
        tuple: The (x, y) coordinates of the middle point 
        """
        x1, y1, x2, y2 = coordinates
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        return (int(x), int(y))
    
    def object_position(self, line_coordinates, object_point):
        """
        Determine the position of the object relative to the line
        Args:
        line_coordinates (list): A list of two tuples [(x1, y1), (x2, y2)] defining the line
        object_point (tuple): The (x, y) coordinates of the object

        Returns:
        str: The position of the object relative to the line ('above' or 'below')
        """
        x1, y1 = line_coordinates[0]
        x2, y2 = line_coordinates[1]
        x_obj, y_obj = object_point

        # Compute the cross product "Determinant"
        cross_product = (x2 - x1) * (y_obj - y1) - (y2 - y1) * (x_obj - x1)

        # Determine the position relative to the line
        if cross_product > 0:
            return "above"
        else:
            return "below"
            
    
    def check_car_position(self, coordinates, camera_number):
        """
        Determine the position of the car relative to the line in camera 2 but it can used for any camera
        Args:
        coordinates (tuple): The (x1, y1, x2, y2) coordinates of the bounding box
        camera_number (int): The camera number

        Returns:
        str: The position of the car relative to the line ('Left' or 'Right')
        """
        if camera_number == 2:
            line_coordinates = [(1050, 380), (1450, 690)]
            position = self.object_position(line_coordinates, self.calculate_middle_point(coordinates))
            if position == "above": # This is must be changed based on the used camera.
                return "Left"
            else:
                return "Right"
    
    def check_person_house_position(self, box_area, point):
        """
        Determine the position of the person relative to the house area in camera 4 but it can used for any camera
        Args:
        box_area (numpy.ndarray): An array of points defining the house area in the form [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        point (tuple): The (x, y) coordinates of the person

        Returns:
        str: The position of the person relative to the house area ('Inside' or 'Outside')
        """
        result = cv2.pointPolygonTest(box_area, (int(point[0]), int(point[1])), False)
        if result >= 0:
            return "Inside"
        elif result < 0:
            return "Outside"
            

    def InitializeInOutHouse(self):
        """
        Initialize the house area for Camera 4
        Returns:
        numpy.ndarray: An array of points defining the house area in the form [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
        """
        points = [(177, 479), (416, 473), (438, 661), (167, 669)]

        # Convert the points to a numpy array
        points_array = np.array(points, dtype=np.int32)

        # Calculate the rotated rectangle from the points
        rect = cv2.minAreaRect(points_array)

        # Get the vertices of the rotated rectangle
        box_area = cv2.boxPoints(rect)
        box_area = np.intp(box_area)

        return box_area

    def is_car(self, obj):
        """
        Check if the object is a car
        Args:
        obj (numpy.ndarray): An array of the form [x1, y1, x2, y2, id, class_id, hit_streak, camera_id] 'single element of TrackResults'
        
        Returns:
        bool: True if the object is a car, False otherwise
        """
        return obj[5] == 0
    
    def is_person(self, obj):
        """
        Check if the object is a person
        Args:
        obj (numpy.ndarray): An array of the form [x1, y1, x2, y2, id, class_id, hit_streak, camera_id] 'single element of TrackResults'
        
        Returns:
        bool: True if the object is a person, False otherwise
        """
        return obj[5] == 1
    
    def process_rtDETR_frame(self, frame, model, image_processor):
        """
        Processes a single frame for object detection.
        Args:
        frame (numpy.ndarray): The frame to process
        model (RTDetrForObjectDetection): The RT-DETR model
        image_processor (RTDetrImageProcessor): The RT-DETR image processor
        
        Returns:
        list: A list of mimic_result objects
        """
        # Save the scales
        width, height = frame.shape[1], frame.shape[0]
        width_scale =  width / 640
        height_scale = height / 640

        frame2 = frame.copy()
        frame2 = cv2.resize(frame2, (640, 640))
        with torch.no_grad():
            inputs = image_processor(images=frame2, return_tensors='pt').to("cuda")
            outputs = model(**inputs)

            # Post-process
            target_sizes = torch.tensor([frame2.shape[:2]]).to("cuda")
            results = image_processor.post_process_object_detection(
                outputs=outputs,
                threshold=0.85,
                target_sizes=target_sizes
            )[0]

            boxes = []

            for box, score, class_id in zip(results["boxes"], results["scores"], results["labels"]):
                box = box.cpu().numpy().astype(int)
                # Multiply the box by the ratio
                box[0] = int(box[0] * width_scale)
                box[1] = int(box[1] * height_scale)
                box[2] = int(box[2] * width_scale)
                box[3] = int(box[3] * height_scale)
                score = score.cpu().numpy()
                class_id = class_id.cpu().numpy()

                # Create a Box object
                boxes.append([box[0], box[1], box[2], box[3], score, class_id-1])

        return [mimic_result(boxes)]


class mimic_result:
    """This class is used to mimic the box object returned by the YOLO model."""
    def __init__(self, lst_of_boxes):
        self.boxes = lst_of_boxes
        

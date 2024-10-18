import cv2
import numpy as np
import keyboard
from ultralytics import YOLO
import time
from sort import Sort


model = YOLO(r"..\runs\detect\train6\weights\best.pt")
classes = model.names

class Camera:
    def __init__(self, number, fps=15):
        self.number = number
        self.name = "Camera " + str(number)
        self.url = f"rtsp://LoginCredentials@IP:Port/Streaming/Channels/{number}01"
        # Send to function called "get_status" to get the status of the camera
        self.status = self.get_status()
        self.fps = fps
    
    
    def get_status(self):
        cap = cv2.VideoCapture(self.url)
        if cap.isOpened():
            cap.release()
            return "Online"
        else:
            cap.release()
            return "Offline"
    
    
    def display(self):
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
    
    def record(self, filename="output.mp4", display=False, skip=0):
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

    def detect(self, write=False, filename="detection.mp4", threshold=0.5, display=True, iou=0.5, draw=True, track=True, Analyse=False, shared_data={}):
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

        Returns:
        None Just displays the frame with/out bounding boxes
        """
        cap = cv2.VideoCapture(self.url)
        
        ret, frame = cap.read()
        skip = False
        frame_time = 1 / 15
        First_Start = time.time()
        frame_counter = 0
        
        if track:
            tracker = Sort(max_age=30, min_hits=5, iou_threshold=0.3)  # Initialize the SORT tracker

        if write:
            output = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*"mp4v"), 15, (frame.shape[1], frame.shape[0]))

        if Analyse and self.number == 4:
            box_area = self.InitializeInOutHouse()

        print("Detecting... Press 'q' to stop detection")
        while True:
            if skip:
                print("Skipping...")
                skip = False
                frame_counter += 1
                continue
            start = time.time()
            ret, frame = cap.read()

            if not ret or (cv2.waitKey(1) & 0xFF == ord('q')):
                break

            results = model(frame, iou=iou, conf=threshold, verbose=False)
            for result in results:
                boxes = result.boxes
                detections = []
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cls = box.cls.item()
                    conf = box.conf.item()
                    detections.append([x1, y1, x2, y2, conf, cls, self.number])
                detections = np.array(detections)
                if track:
                    TrackResults = self.tracking(detections, tracker, shared_data)

                    if draw and Analyse: # Draw and Analyse
                        if self.number == 2:
                            # Draw the line
                            cv2.line(frame, (80, 0), (1280, 890), (0, 0, 255), 2)
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

            elapsed_time = time.time() - First_Start
            # print("Processing time for this frame", time.time() - start)
            # print(elapsed_time,"elapsed time for frame", frame_counter + 1)

            # Calculate the time at which the next frame should be displayed
            target_time = (frame_counter + 1) * frame_time
            threshold_time = (frame_counter + 90) * frame_time
            # print("threshold_time", threshold_time, "for the frame", frame_counter + 90)
            if elapsed_time > threshold_time:
                skip = True

            
            wait_time = target_time - elapsed_time
            # If wait_time is positive, sleep for that duration
            if wait_time > 0:
                time.sleep(wait_time)

            frame_counter += 1

            
        cap.release()
        cv2.destroyAllWindows()
    
    def tracking(self, detections, tracker, shared_data={}):
        if detections.any() == True:
            TrackResults = tracker.update(detections[:, :7], shared_data)  # Pass the bounding box and confidence score
        else:
            TrackResults = tracker.update(np.empty((0, 7)), shared_data)
        
        return TrackResults
    
    def Analyse(self, tracker, TrackResults, shared_data, box_area=None):
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
        x1, y1, x2, y2 = obj[:4]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)
        cv2.putText(frame, classes[int(obj[5])], (x1, y1 + 20), font, 0.5, class_color, thickness)
        if with_id:
            id = int(obj[4]) 
            cv2.putText(frame, f"ID: {id}", (x1, y1 - 10), font, 0.5, id_color, thickness)

        return frame
        

    
    def calculate_middle_point(self, coordinates):
        x1, y1, x2, y2 = coordinates
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2
        return (int(x), int(y))
    
    def object_position(self, line_coordinates, object_point):
        """
        Determines the relative position of an object to a line defined by start_point and end_point.
        
        Args:
            start_point (tuple): The (x, y) coordinates of the line's starting point.
            end_point (tuple): The (x, y) coordinates of the line's end point.
            object_point (tuple): The (x, y) coordinates of the object.
            
        Returns:
            str: A string indicating whether the object is 'above', 'below'.
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
            
    
    def check_car_position(self, coordinates, camera_number): # Camera 2
        if camera_number == 2:
            line_coordinates = [(80, 0), (1280, 890)]
            position = self.object_position(line_coordinates, self.calculate_middle_point(coordinates))
            if position == "above": # This is must be changed based on the used camera.
                return "Left"
            else:
                return "Right"
    
    def check_person_house_position(self, box_area, point): # Camera 4
        result = cv2.pointPolygonTest(box_area, (int(point[0]), int(point[1])), False)
        if result >= 0:
            return "Inside"
        elif result < 0:
            return "Outside"
            

    def InitializeInOutHouse(self):
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
        return obj[5] == 0
    
    def is_person(self, obj):
        return obj[5] == 1
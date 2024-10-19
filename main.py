 import time
import math
import multiprocessing
import sys
import os
import torch
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Classes import Camera
import requests

def process_camera(camera, shared_data, write, threshold, display, iou, draw, track, Analyse):
    filename = f"detection_output_{camera.number}.mp4"
    camera.detect(write, filename, threshold, display, iou, draw, track, Analyse, shared_data)
    print(shared_data)

def send_push_notification(message):
    url = 'https://api.pushover.net/1/messages.json'
    payload = {
        'token': 'TokenID',  # Replace with your Pushover application token
        'user': 'UserID',    # Replace with your Pushover user key
        'message': message          # The message you want to send
    }
    response = requests.post(url, data=payload)
    print('Notification sent, status:', response.status_code)

def monitor_shared_data(shared_data):
    notified_outside = False  # Flag to track notification for "Right Now Outside"
    notified_entered = False  # Flag to track notification for "Entered The House"
    

    last_notified_left_to_right = 0  # Track last notified count for left to right
    last_notified_right_to_left = 0  # Track last notified count for right to left
    last_left_the_house = 0

    while True:
        current_left_to_right = shared_data.get("Left_to_right", 0)
        current_right_to_left = shared_data.get("Right_to_left", 0)
        current_leave = shared_data.get("Left The House", 0)

        # Someone left the house
        if current_leave > last_left_the_house:
            send_push_notification("Someone has left the house.")
            last_left_the_house = current_leave

        if shared_data.get("Right Now Outside", 0) > 5 and not notified_outside:
            send_push_notification("More than 5 people are currently outside the house. Please check the situation.")
            notified_outside = True  # Set the flag to True after sending notification
        elif shared_data.get("Right Now Outside", 0) <= 5:
            notified_outside = False  # Reset the flag when the condition no longer holds

        if shared_data.get("Entered The House", 0) > 10 and not notified_entered:
            send_push_notification("More than 10 people have entered the house. This might be unusual.")
            notified_entered = True  # Set the flag to True after sending notification

        elif shared_data.get("Entered The House", 0) <= 10:
            notified_entered = False  # Reset the flag when the condition no longer holds

        if current_right_to_left // 5 > last_notified_right_to_left // 5:
            send_push_notification("A new 5 cars have moved from right to left.")
            last_notified_right_to_left = current_right_to_left  # Update last notified count
        
        if current_left_to_right // 5 > last_notified_left_to_right // 5:
            send_push_notification("A new 5 cars have moved from left to right.")
            last_notified_left_to_right = current_left_to_right  # Update last notified count
        
        time.sleep(0.2)  # Poll every half second


def json_serializable(data):
    if isinstance(data, multiprocessing.managers.ListProxy):
        return list(data)
    return data

if __name__ == "__main__":
    camera1 = Camera(1)
    camera2 = Camera(2)
    camera3 = Camera(3)
    camera4 = Camera(4)
    camera6 = Camera(6)
    numbers = [camera1, camera2, camera3, camera4, camera6]

    print(torch.cuda.is_available())

    # Create a Manager to share data between processes
    with multiprocessing.Manager() as manager:
        shared_data = manager.dict()
        shared_data["logs"] = manager.list()

        # Start the monitoring process
        monitor_process = multiprocessing.Process(target=monitor_shared_data, args=(shared_data,))
        monitor_process.start()

        # Create a Pool with the number of available CPU cores
        pool = multiprocessing.Pool(processes=5)
        print(multiprocessing.cpu_count())
        
        write = True
        threshold = 0.6
        display = True
        iou = 0.4
        draw = True
        track = True
        Analyse = True

        if Analyse:
            shared_data["Right_to_left"] = 0
            shared_data["Left_to_right"] = 0
            shared_data["Entered The House"] = 0
            shared_data["Left The House"] = 0
            shared_data["Right Now Inside"] = 1
            shared_data["Right Now Outside"] = 0
            shared_data["Is car 1 at the house?"] = None
            shared_data["Is car 2 at the house?"] = None

        # Use apply_async for asynchronous processing and pass the arguments
        for camera in numbers:
            pool.apply_async(process_camera, args=(camera, shared_data, write, threshold, display, iou, draw, track, Analyse))

        # Close the pool and wait for the processes to complete
        pool.close()
        pool.join()

        # Terminate the monitoring process
        monitor_process.terminate()
        monitor_process.join()

        # Convert ListProxy to a regular list for serialization
        if 'logs' in shared_data:
            shared_data['logs'] = list(shared_data['logs'])

        # Save the shared data
        try:
            with open("shared_data.json", "w") as json_file:
                json.dump({key: json_serializable(value) for key, value in shared_data.items()}, json_file, indent=4)
        except IOError as e:
            print(f"An error occurred while writing to file: {e}")

        print("\nAll tasks completed.")

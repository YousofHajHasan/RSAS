import time
import multiprocessing
import sys
import os
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from Classes import Camera
import requests
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def initialize_scene(ax):
    """Initialize the static elements of the scene.
    Args:
    ax (Axes3D): The 3D axes object
    Returns:
    None
    """
    # Hide all axes
    ax.axis('off')

    # Changes the main background
    ax.set_facecolor('black')  
    
    # Set up the 3D space limits
    ax.set_xlim(0, 350)
    ax.set_ylim(0, 500)  # Depth
    ax.set_zlim(0, 100)  # Height

    # Plot the road (flat, no height)
    road_vertices = [
        [0, 10, 0], [350, 10, 0], [350, 180, 0], [0, 180, 0]
    ]
    road_poly = Poly3DCollection([road_vertices], color='gray', alpha=0.5)
    ax.add_collection3d(road_poly)

    # Dashed centerline for the road
    for x in range(0, 350, 20):
        ax.plot([x, x + 10], [95, 95], [0, 0], color='white', linewidth=1)

    # Plot the house (black)
    house_base = [
        [70, 250, 0], [290, 250, 0], [290, 410, 0], [70, 410, 0]
    ]
    house_top = [
        [70, 250, 40], [290, 250, 40], [290, 410, 40], [70, 410, 40]
    ]

    sides = [
        [house_base[0], house_base[1], house_top[1], house_top[0]],  # Front
        [house_base[1], house_base[2], house_top[2], house_top[1]],  # Right
        [house_base[2], house_base[3], house_top[3], house_top[2]],  # Back
        [house_base[3], house_base[0], house_top[0], house_top[3]]   # Left
    ]

    for side in sides:
        side_poly = Poly3DCollection([side], color='white', alpha=0.4)
        ax.add_collection3d(side_poly)

    roof_poly = Poly3DCollection([house_top], color='white', alpha=0.4)
    ax.add_collection3d(roof_poly)

    # Plot the boundaries (height = 10)
    boundary_base_1 = [[0, 210, 0], [160, 210, 0], [160, 210, 10], [0, 210, 10]]  # Left part of front boundary
    boundary_base_2 = [[200, 210, 0], [350, 210, 0], [350, 210, 10], [200, 210, 10]]  # Right part of front boundary
    boundary_base_3 = [[0, 500, 0], [350, 500, 0], [350, 500, 10], [0, 500, 10]]  # Back boundary
    boundary_side_1 = [[0, 210, 0], [0, 500, 0], [0, 500, 10], [0, 210, 10]]  # Left boundary
    boundary_side_2 = [[350, 210, 0], [350, 500, 0], [350, 500, 10], [350, 210, 10]]  # Right boundary

    boundaries = [boundary_base_1, boundary_base_2, boundary_base_3, boundary_side_1, boundary_side_2]

    for boundary in boundaries:
        boundary_poly = Poly3DCollection([boundary], color=[0.4, 0.4, 0.4, 1])
        ax.add_collection3d(boundary_poly)
    
    # Add door boundary
    # Door boundary in the space between left and right front boundaries
    boundary_door = [
    # Bottom-left corner of the rectangle at ground level
    [200, 210, 0] ,
    # Bottom-right corner of the rectangle at ground level
    [160, 210, 0],
    # Top-right corner of the rectangle at a height of 10 units
    [160, 210, 15],
    # Top-left corner of the rectangle at a height of 10 units
    [200, 210, 15]
    ]

    door_poly = Poly3DCollection([boundary_door], color='white', alpha=0.6)
    ax.add_collection3d(door_poly)

    # Add house windows
    left_window = [105, 240, 15], [145, 240, 15], [145, 240, 30], [105, 240, 30]
    right_window = [215, 240, 15], [255, 240, 15], [255, 240, 30], [215, 240, 30]

    windows = [left_window, right_window]

    for window in windows:
        window_poly = Poly3DCollection([window], color='white', alpha=0.6)
        ax.add_collection3d(window_poly)

    # Add sidewalk
    sidewalk = [
        [0, 180, 0], [350, 180, 0], [350, 210, 3], [0, 210, 3]
    ]
    sidewalk_poly = Poly3DCollection([sidewalk], color='red', alpha=0.4)
    ax.add_collection3d(sidewalk_poly)

    # Add labels
    ax.text(160, 230, 70, 'House', color='Yellow', fontsize=12, weight='heavy')
    ax.text(300, 230, 35, '1st Garage', color='Yellow', fontsize=12, weight='heavy')
    ax.text(0, 230, 35, '2nd Garage', color='Yellow', fontsize=12, weight='heavy')

    # Add the green backyard
    backyard = [
        [0, 430, 0], [160, 430, 0], [160, 500, 0], [0, 500, 0]
    ]
    backyard_poly = Poly3DCollection([backyard], color='green', alpha=0.4)
    ax.add_collection3d(backyard_poly)


def plot_moving_object(ax, object_details, cube_artists, with_tracking):
    """Plot or update the moving cube.
    Args:
    ax (Axes3D): The 3D axes object
    object_details (list): The details of the object to plot
    cube_artists (list): A list to store the cube artists
    with_tracking (bool): Whether to add tracking ID on the cube or not
    Returns:
    None
    """
    cube_position = object_details[:2]
    Track_id = object_details[2]
    object_class = object_details[3]
    # Define the cube geometry
    if object_class == "person":
        cube_base = [
            [cube_position[0], cube_position[1], 0],
            [cube_position[0] + 5, cube_position[1], 0],
            [cube_position[0] + 5, cube_position[1] + 5, 0],
            [cube_position[0], cube_position[1] + 5, 0]
        ]
        cube_top = [
            [cube_position[0], cube_position[1], 10],
            [cube_position[0] + 5, cube_position[1], 10],
            [cube_position[0] + 5, cube_position[1] + 5, 10],
            [cube_position[0], cube_position[1] + 5, 10]
        ]

        cube_sides = [
            [cube_base[0], cube_base[1], cube_top[1], cube_top[0]],  # Front
            [cube_base[1], cube_base[2], cube_top[2], cube_top[1]],  # Right
            [cube_base[2], cube_base[3], cube_top[3], cube_top[2]],  # Back
            [cube_base[3], cube_base[0], cube_top[0], cube_top[3]]   # Left
        ]

        # Add the cube sides and top
        for side in cube_sides:
            cube_poly = Poly3DCollection([side], color='green', alpha=1)
            ax.add_collection3d(cube_poly)
            cube_artists.append(cube_poly)

        cube_top_poly = Poly3DCollection([cube_top], color='green', alpha=1)
        ax.add_collection3d(cube_top_poly)
        cube_artists.append(cube_top_poly)
    
    elif object_class == "car":
        cube_base = [
            [cube_position[0], cube_position[1], 0],
            [cube_position[0] + 20, cube_position[1], 0],
            [cube_position[0] + 20, cube_position[1] + 10, 0],
            [cube_position[0], cube_position[1] + 10, 0]
        ]
        cube_top = [
            [cube_position[0], cube_position[1], 10],
            [cube_position[0] + 20, cube_position[1], 10],
            [cube_position[0] + 20, cube_position[1] + 10, 10],
            [cube_position[0], cube_position[1] + 10, 10]
        ]

        cube_sides = [
            [cube_base[0], cube_base[1], cube_top[1], cube_top[0]],  # Front
            [cube_base[1], cube_base[2], cube_top[2], cube_top[1]],  # Right
            [cube_base[2], cube_base[3], cube_top[3], cube_top[2]],  # Back
            [cube_base[3], cube_base[0], cube_top[0], cube_top[3]]   # Left
        ]

        # Add the cube sides and top
        for side in cube_sides:
            cube_poly = Poly3DCollection([side], color='red', alpha=1)
            ax.add_collection3d(cube_poly)
            cube_artists.append(cube_poly)

        cube_top_poly = Poly3DCollection([cube_top], color='red', alpha=1)
        ax.add_collection3d(cube_top_poly)
        cube_artists.append(cube_top_poly)

    if with_tracking:
        # Add track ID on the top of the cube
        text = ax.text(cube_position[0] + 10, cube_position[1] + 5, 10, f"ID: {int(Track_id)}", color='white', fontsize=8, weight='heavy')
        cube_artists.append(text)
    

def clear_lists(ax, cubes, cube_artists):
    """
    This function will be called by the Visualize process to clear the lists every time before plotting the new cubes.
    Args:
    ax (Axes3D): The 3D axes object
    cubes (list): A list to store the cube positions [x, y, Track_id, object_class], Track_id could be None
    cube_artists (list): A list to store the cube artists (Poly3DCollection and Text)
    Returns:
    None
    """
    if cube_artists:
        for artist in cube_artists:
            artist.remove()

    cubes[:] = []
    cube_artists.clear()

def Visualize(cubes, cube_artists, flags):
    """
    Works as a manager for the 3D projection of the moving objects.
    Args:
    cubes (list): A list to store the cube positions [x, y, Track_id, object_class], Track_id could be None
    cube_artists (list): A list to store the cube artists (Poly3DCollection and Text)
    flags (dict): A multiprocessing manager dictionary to control the visualization process, 
    each camera will have a flag to control the visualization in addition to a general flag for the Visualizer process.
    Returns:
    None
    """

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    initialize_scene(ax)

    while True:
        # Change the flag to True to indicate that the Visualizer is running, so the other processors will not update the lists.
        flags["Visualizer"].value = True

        # I forget why did I add this line, but I think it's better to keep it. I should've added a comment. But I think it's something related to multiprocessing.
        not_shared_cubes = list(cubes)

        start_time = time.time()

        clear_lists(ax, cubes, cube_artists)
        for object_details in not_shared_cubes:
            if object_details[2] is not None: # If the object has a tracking ID
                plot_moving_object(ax, object_details, cube_artists, True)
            else:
                plot_moving_object(ax, object_details, cube_artists, False)

        elapsed_time = time.time() - start_time # This measures the time taken to plot the cubes.
        
        # Calculate remaining time to sleep
        sleep_time = max(0.0001, 0.2 - elapsed_time) # 0.2 seconds is the refresh rate for the visualization. 
        # If the time taken to plot the cubes is more than 0.2 seconds, then sleep_time will be 0.0001 seconds to avoid the delay.
        for key, value in flags.items():
            flags[key].value = False # Reset the flags to False, so the other processors can update the lists.
        plt.pause(sleep_time) # Preview the plot for sleep_time seconds.
        # If the scene is closed, break the loop
        if not plt.fignum_exists(fig.number):
            break


def process_camera(camera, shared_data, write, threshold, display, iou, draw, track, Analyse, projection, flags, cubes):
    """
    Process the camera stream for object detection
    args:
    camera (Camera): Camera object
    shared_data (dict): A dictionary to store shared data between processes
    write (bool): whether to write the output to a file or not
    threshold (float): Confidence threshold
    display (bool): whether to display the output or not
    iou (float): Intersection over Union threshold
    draw (bool): whether to draw the bounding boxes on the frame or not
    track (bool): whether to track the objects in the frame or not
    Analyse (bool): whether to analyse the objects in the frame or not
    projection (bool): whether to project the objects in 3D space or not
    flags (dict): A multiprocessing manager dictionary to control the visualization process,
    cubes (list): A list to store the cube positions [x, y, Track_id, object_class].

    Returns:
    None 'Just displays the frame with/out bounding boxes and prints the shared data'
    """
    
    print(f"CUDA available in process: {torch.cuda.is_available()}")
    print(camera.get_status())
    filename = f"detection_output_{camera.number}.mp4"
    camera.detect(write, filename, threshold, display, iou, draw, track, Analyse, projection, flags, cubes, shared_data)
    print(shared_data)

def send_push_notification(message):
    """
    Send a push notification using the Pushover API
    Args:
    message (str): The message to send
    Returns:
    None
    """
    url = 'https://api.pushover.net/1/messages.json'
    payload = {
        'token': 'ajjv4p19kj75cwzghttab85u1vy47w',  # Replace with your Pushover application token
        'user': 'uuye2u9iuy8kjke6rmodrpfvmvmzag',    # Replace with your Pushover user key
        'message': message          # The message you want to send
    }
    response = requests.post(url, data=payload)
    print('Notification sent, status:', response.status_code)

def monitor_shared_data(shared_data):
    """
    Monitor the shared data for any anomalies and send push notifications
    Args:
    shared_data (dict): A dictionary to store shared data between processes
    Returns:
    None 'Just modifies the shared data and sends push notifications'
    """
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
    """
    Convert the data to a JSON serializable format because multiprocessing.Manager does not support all data types
    Args:
    data: The data to convert in form of a ListProxy
    Returns:
    The converted data in a regular list
    """
    if isinstance(data, multiprocessing.managers.ListProxy):
        return list(data)
    return data

if __name__ == "__main__":
    # Set the start method to 'spawn' to avoid errors with multiprocessing on Linux.
    multiprocessing.set_start_method('spawn')

    camera1 = Camera(1)
    camera2 = Camera(2)
    camera3 = Camera(3)

    Cameras = [camera1, camera2, camera3]

    # Create a Manager to share data between processes
    with multiprocessing.Manager() as manager:
        shared_data = manager.dict()
        shared_data["logs"] = manager.list()
        cubes = manager.list()
        cube_artists = []
        flags = manager.dict()

        # Initialize the flags 
        flags["Visualizer"] = manager.Value('b', False)
        for camera in Cameras:
            # These flags are used to control the position addition for each second per processor.
            flags[camera.number] = manager.Value('b', True)


        # Start the monitoring process in a separate process
        monitor_process = multiprocessing.Process(target=monitor_shared_data, args=(shared_data,))
        monitor_process.start()

        # Set the arguments for the process_camera function
        write = False
        threshold = 0.5
        display = True
        iou = 0.8
        draw = True
        track = True
        Analyse = True
        projection = True

        # Set the initial values for the shared data (if Analyse is enabled)
        if Analyse:
            shared_data["Right_to_left"] = 0
            shared_data["Left_to_right"] = 0
            shared_data["Entered The House"] = 0
            shared_data["Left The House"] = 0
            shared_data["Right Now Inside"] = 1
            shared_data["Right Now Outside"] = 0
            shared_data["Is car 1 at the house?"] = None
            shared_data["Is car 2 at the house?"] = None

        # Create and start processes for each camera
        processes = []
        for camera in Cameras:
            process = multiprocessing.Process(target=process_camera, args=(camera, shared_data, write, threshold, display, iou, draw, track, Analyse, projection, flags, cubes))
            processes.append(process)
            process.start()

        if projection:
            time.sleep(1)
            # Start the Visualize processor
            Visualize_Process = multiprocessing.Process(target=Visualize, args=(cubes, cube_artists, flags))
            Visualize_Process.start()
            processes.append(Visualize_Process)

        # Wait for the camera processes to complete
        for process in processes:
            process.join()

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

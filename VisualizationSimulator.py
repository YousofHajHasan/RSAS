import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import time
import multiprocessing


def processor_task(cubes, processor_id):
    """Function executed by each processor to add values to the shared list."""
    for i in range(30):
        # Run every 1 second
        time.sleep(1)
        Seconds = time.strftime("%H:%M:%S", time.localtime())[-2:]
        cube_position = [70 * processor_id, i + int(Seconds)]
        if not flag.value:
            cubes.append(cube_position)
            print(f"Processor {processor_id} added a new position")
        else:
            print(f"Processor {processor_id} skipped adding due to visualization.")


def initialize_scene(ax):
    """Initialize the static elements of the scene."""
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


def plot_moving_object(ax, cube_position, cube_artists):
    """Plot or update the moving cube."""
    # Define the cube geometry
    cube_base = [
        [cube_position[0], cube_position[1], 0],
        [cube_position[0] + 10, cube_position[1], 0],
        [cube_position[0] + 10, cube_position[1] + 10, 0],
        [cube_position[0], cube_position[1] + 10, 0]
    ]
    cube_top = [
        [cube_position[0], cube_position[1], 10],
        [cube_position[0] + 10, cube_position[1], 10],
        [cube_position[0] + 10, cube_position[1] + 10, 10],
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
        cube_poly = Poly3DCollection([side], color='red', alpha=0.8)
        ax.add_collection3d(cube_poly)
        cube_artists.append(cube_poly)

    cube_top_poly = Poly3DCollection([cube_top], color='red', alpha=0.8)
    ax.add_collection3d(cube_top_poly)
    cube_artists.append(cube_top_poly)


def clear_lists(ax, cubes, cube_artists):
    """Clear the shared list if allowed."""
    print("Clearing process running...")
    if cube_artists:
        for artist in cube_artists:
            artist.remove()

    cubes[:] = []
    cube_artists.clear()

def Visualize(cubes, cube_artists):
    print("Entered Visualize")

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    initialize_scene(ax)

    while True:    
        
        print("Switching flag to True")
        flag.value = True

        # No need for reading, both lists are being updated by the other processors simultaneously.
        # OR I think I need
        not_shared_cubes = list(cubes)
        

        start_time = time.time()  # Record the start time
        # Simulate the cube's movement
        print(not_shared_cubes)
        print("cube artists",len(cube_artists))
        clear_lists(ax, cubes, cube_artists)
        for cube_position in not_shared_cubes:
            plot_moving_object(ax, cube_position, cube_artists)  # Add/update the cube
            print("Finished 1 plot_moving_object")
        print("Finished plot_moving_object")
        elapsed_time = time.time() - start_time  # Measure the elapsed time
        
        # Calculate remaining time to sleep
        sleep_time = max(0.0001, 1 - elapsed_time) # 0 Will cause the plot to freeze and wait for q to close the window
        print("items to visualize:", len(cube_artists))
        print("plt.pause for ", sleep_time)
        print("switching flag to False")
        flag.value = False
        plt.pause(sleep_time)

        
        



if __name__ == "__main__":

    with multiprocessing.Manager() as manager:
        cubes = manager.list()
        cube_artists = []
        flag = multiprocessing.Value('b', False)
        

        # Create and start "adding" processors
        num_processors = 4
        processes = []
        for i in range(num_processors):
            process = multiprocessing.Process(target=processor_task, args=(cubes, i))
            processes.append(process)
            process.start()

        time.sleep(1)
        # Start the Visualize processor
        Visualize_Process = multiprocessing.Process(target=Visualize, args=(cubes, cube_artists))
        Visualize_Process.start()
        processes.append(Visualize_Process)

        # Wait for "adding" processors to finish
        for process in processes:
            process.join()
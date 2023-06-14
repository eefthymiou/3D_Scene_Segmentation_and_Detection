import numpy as np
import cv2
import open3d as o3d
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


#constants
red = np.array([1,0,0])
green = np.array([0,1,0])
blue = np.array([0,0,1])
cyan = np.array([0,1,1])
yellow = np.array([1,1,0])
magenta = np.array([1,0,1])
black = np.array([0,0,0])
white = np.array([1,1,1])



def disp_image_and_rectangle(img, rect_start, template_rows, template_cols):
    # Display the original image
    plt.imshow(img, cmap='gray')

    # Get the current reference
    ax = plt.gca()

    # Create a Rectangle patch
    rect = Rectangle(rect_start, template_cols, template_rows, linewidth=1, edgecolor='r', facecolor='none')

    # Add the patch to the Axes
    ax.add_patch(rect)

    plt.show()

def visualize_stereo_frames(left_frame, right_frame):
    # Concatenate the left and right frames horizontally
    concatenated_frames = np.hstack((left_frame, right_frame))

    # Convert to RGB if the input is in BGR format
    if left_frame.ndim == 3 and left_frame.shape[2] == 3:
        concatenated_frames = cv2.cvtColor(concatenated_frames, cv2.COLOR_BGR2RGB)

    # Display the concatenated frames using Matplotlib
    plt.figure(figsize=(12, 5))
    plt.imshow(concatenated_frames, cmap='gray' if left_frame.ndim == 2 else None)
    plt.axis('off')
    plt.show()

def visualize_disparity_map(disparity_map, cmap='jet'):
    # Normalize the disparity map for visualization
    #normalized_disparity = (disparity_map - disparity_map.min()) / (disparity_map.max() - disparity_map.min())
    normalized_disparity = disparity_map
    # Display the disparity map using Matplotlib
    plt.figure(figsize=(12, 5))
    plt.imshow(normalized_disparity, cmap=cmap)
    plt.colorbar()
    plt.axis('off')
    plt.show()

def visualize_point_cloud(pc, colors=None):
    print("start visualization")
    pc = o3d.utility.Vector3dVector(pc)
    pc = o3d.geometry.PointCloud(pc)

    if colors is not None:
        if colors.shape[-1] == 3:
            print("RGB")
            pc.colors = o3d.utility.Vector3dVector(colors)

    vis = o3d.visualization.Visualizer()  # Create a visualizer object
    vis.create_window()  # Create a window for visualization
    vis.add_geometry(pc)  # Add the point cloud to the visualizer
    vis.run()

    while True:
        # Check for keyboard event
        if vis.poll_events():
            break

    # Close the window
    vis.destroy_window()
    print("end visualization")

# get the center of the poincloud
def get_center(pc):
    vertices = np.asarray(pc.points)
    center = np.mean(vertices, axis=0)
    return center

# unit sphere normalization
def unit_sphere_normalization(pc):
    vertices = np.asarray(pc.points)
    distance = np.sqrt(((vertices * vertices).sum(axis = -1)))
    max_distance = np.max(distance)
    vertices /= max_distance
    pc.points = o3d.utility.Vector3dVector(vertices)
    return pc

# translate for pc
def translate(pc, translation_vec):
    vertices = np.asarray(pc.points)
    vertices += translation_vec
    pc.points = o3d.utility.Vector3dVector(vertices)
    return pc

# translate for mesh
def translate_mesh(mesh, translation_vec):
    vertices = np.asarray(mesh.vertices)
    vertices += translation_vec
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return mesh
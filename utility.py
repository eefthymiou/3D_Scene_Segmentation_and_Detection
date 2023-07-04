import numpy as np
import cv2
import open3d as o3d
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from scipy.spatial import ConvexHull
import os


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
    vertices *= 1000
    pc.points = o3d.utility.Vector3dVector(vertices)
    return pc

def unit_sphere_normalization_vertices(vertices):
    distance = np.sqrt(((vertices * vertices).sum(axis = -1)))
    max_distance = np.max(distance)
    vertices /= (max_distance)
    vertices *= 1000
    return vertices

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

# translate for lineset
def translate_LineSet(lineset, translation_vec):
    vertices = np.asarray(lineset.points)
    vertices += translation_vec
    lineset.points = o3d.utility.Vector3dVector(vertices)
    return lineset


def chull(vertices):
    # udentify if the point cloud is 2D or 3D

    # get the min and max of the z axis
    min_z = np.min(vertices[:,2])
    max_z = np.max(vertices[:,2])

    # if the min and max are equal then the point cloud is 2D
    if min_z == max_z:
        return chull_2D_pc(vertices)
    else:
        return chull_3D_pc(vertices)
    

def chull_3D_pc(vertices):
    # compute the convex hull
    hull = ConvexHull(vertices)
    
    # get the faces of the convex hull
    hull_faces = hull.simplices

    hull_mesh = o3d.geometry.TriangleMesh()
    hull_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    hull_mesh.triangles = o3d.utility.Vector3iVector(hull_faces)

    return hull_mesh

def chull_2D_pc(vertices):
    # compute the convex hull
    # keep the z value 
    z = vertices[0,2]
    vertices = np.asarray(vertices)[:,:2]
    hull = ConvexHull(vertices)
    #convert the convex hull to a lineset
    hull = chull_to_lineset(vertices[hull.vertices],z)

    return hull

#Converts a 2d set of points representing the convex hull to a lineset
#assumes the points are sorted
def chull_to_lineset(points,z):
    points = pad_2d(points, None, -1, 0)

    indices1 = np.arange(points.shape[0])
    indices2 = np.arange(points.shape[0])+1
    indices2[-1] = 0

    indices = np.vstack((indices1, indices2)).T

    # set the z value
    points[:,-1] = z

    return o3d.geometry.LineSet(
        o3d.utility.Vector3dVector(points),
        o3d.utility.Vector2iVector(indices)
    )

#utility function for padding a 2d array with a row and/or column of a specific value
def pad_2d(arr, row=None, col=None, value=0):

    '''
        arr: np.array with shape (N, M)
    '''

    if len(arr.shape) != 2:
        raise ValueError(f"Only 2d numpy arrays are accepted. You gave {len(arr.shape)}d")

    if (not isinstance(row, int)) and (row is not None):
        raise ValueError("Only integer values -1 <= row <= num_rows are accepted")
    
    if (not isinstance(col, int)) and (col is not None):
        raise ValueError("Only integer values -1 <= row <= num_cols are accepted")

    #trivial case, no change is required
    if row is None and col is None:
        return arr

        
    zero_row = np.ones((1, arr.shape[1])) * value

    if row is not None:
        if row < -1 or row > arr.shape[0]:
            raise ValueError("Only integer values -1 <= row <= num_rows are accepted")
        elif row > 0:
            arr = np.concatenate(
                (arr[:row,:], zero_row, arr[row:, :]), axis=0
            )
        elif row == -1:
            arr = np.concatenate(
                (arr, zero_row), axis=0
            )

    zero_col = np.ones((arr.shape[0], 1)) * value

    if col is not None:
        if col < -1 or col > arr.shape[1]:
            raise ValueError("Only integer values -1 <= row <= num_rows are accepted")
        elif col > 0:
            arr = np.concatenate(
                (arr[:,:col], zero_col, arr[:, col:]), axis=1
            )
        elif col == -1:
            arr = np.concatenate(
                (arr, zero_col), axis=1
            )

    return arr

#returns an open3d point cloud from a point array
def o3d_pointcloud(verts, center=False, color=black):

    assert verts.shape[-1] in (2, 3)
    assert len(verts.shape) == 2

    if verts.shape[-1] == 2:
        verts = pad_2d(verts, row = None, col=-1, value=0)

    if center:
        centroid = verts.mean(0)
        verts = verts - centroid

    if len(color.shape) == 1:
        return o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(verts)
        ).paint_uniform_color(color)
    else:
        assert len(color.shape) == 2
        assert color.shape[0] == verts.shape[0]

        pcloud = o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(verts)
        )

        pcloud.colors = o3d.utility.Vector3dVector(color)
        return pcloud

def count_directories(path):
    count = 0
    for item in os.listdir(path):
        item_path = os.path.join(path, item)
        if os.path.isdir(item_path):
            count += 1
    return count

def downsample(point_cloud, a):

    points = np.asarray(point_cloud.points)
    N = np.shape(points)[0]

    indices = np.arange(N)
    M = N // a
    indices = np.random.choice(indices, M, replace = False)

    points = points[indices,:]
    colors = np.asarray(point_cloud.colors)[indices,:]

    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)
    return point_cloud, M


def plane_from_points(x1, x2, x3):
    AB = np.array(x2) - np.array(x1)
    AC = np.array(x3) - np.array(x1)
    N = np.cross(AB, AC)
    d = -np.dot(N, np.array(x1))
    return N, d

def closest_point_on_line(point, line_start, line_end):
    # Calculate the closest point on a line to a given point
    line_vec = np.array(line_end) - np.array(line_start)
    point_vec = np.array(point) - np.array(line_start)
    line_length = np.linalg.norm(line_vec)
    line_unit_vec = line_vec / line_length
    t = np.dot(line_unit_vec, point_vec)
    t = max(0, min(t, line_length))
    closest_point = np.array(line_start) + t * line_unit_vec
    return closest_point

def distance(point1, point2):
    # Calculate the Euclidean distance between two points in 3D space
    return np.linalg.norm(np.array(point1) - np.array(point2))

def find_AABB(points):
    points = np.asarray(points)

    minxyz = np.min(points,axis=0)
    maxxyz = np.max(points,axis=0)

    aabb = o3d.geometry.AxisAlignedBoundingBox(minxyz,maxxyz)
    aabb.color = black
    return aabb


def is_sphere_inside_bounding_box(center, radius, bbox):
    # Get the minimum and maximum points of the bounding box
    bbox_min = np.array(bbox.min_bound)
    bbox_max = np.array(bbox.max_bound)

    # Calculate the closest point on the bounding box to the sphere's center
    closest_point = np.clip(center, bbox_min, bbox_max)

    # Calculate the distance between the sphere's center and the closest point on the bounding box
    distance = np.linalg.norm(center - closest_point)

    # Check if the distance is less than or equal to the sphere's radius
    is_inside = distance <= radius

    return is_inside
        


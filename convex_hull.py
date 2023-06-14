
import numpy as np
import open3d as o3d

from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt

import os

import part_b.utility_b as U

# import point cloud
vertices = np.load('pointcloud/gt_objects_clustered.npy')
colors = np.load('pointcloud/gt_objects_clustered.npy')

hull_mesh = U.chull_3D_pc(vertices)

directory = 'clusters/gt_clusters/cluster_4'
# Save the remaining point cloud colors as a NPY files
# Save the convex hull as an OBJ file
o3d.io.write_triangle_mesh(os.path.join(directory, "convex_hull.obj"), hull_mesh)

# Save the remaining point cloud colors as a NPY files
np.save(os.path.join(directory, "remaining_points.npy"), vertices)
    





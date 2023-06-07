import numpy as np
import utility as U

def main():
    # part_a paths to point cloud
    vertices_path = './part_a/pointcloud/vertices.npy'
    colors_path = './part_a/pointcloud/colors.npy'

    # part_b paths to point cloud
    # vertices_path = './part_b/pointcloud/gt_vertices.npy'
    # colors_path = './part_b/pointcloud/gt_colors.npy'

    # load vertices and colors
    vertices = np.load(vertices_path)
    colors = np.load(colors_path)

    # visualize point cloud
    U.visualize_point_cloud(vertices, colors)

main()
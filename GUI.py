import numpy as np
import utility as U

def main():
    # pointcloud paths
    vertices_path = 'pointcloud/vertices.npy'
    colors_path = 'pointcloud/colors.npy'

    # gt_pointcloud paths
    vertices_path = 'pointcloud/gt_vertices.npy'
    colors_path = 'pointcloud/gt_colors.npy'

    # load inliers
    vertices_path = 'pointcloud/inlier_vertices.npy'
    colors_path = 'pointcloud/inlier_colors.npy'

    # load outliers
    # vertices_path = 'pointcloud/outlier_vertices.npy'
    # colors_path = 'pointcloud/outlier_colors.npy'

    # load vertices and colors
    vertices = np.load(vertices_path)
    colors = np.load(colors_path)

    # visualize point cloud
    U.visualize_point_cloud(vertices, colors)

main()
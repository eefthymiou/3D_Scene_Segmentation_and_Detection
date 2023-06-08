import numpy as np
import utility as U

def main():
    # 1 -> my pointcloud
    # 2 -> gt pointcloud
    # 3 -> inliers (plane)
    # 4 -> outliers (objects)

    select_poincloud = 1

    if select_poincloud == 0:
        # pointcloud paths
        vertices_path = 'pointcloud/vertices.npy'
        colors_path = 'pointcloud/colors.npy'

    elif select_poincloud == 1:
        # gt_pointcloud paths
        vertices_path = 'pointcloud/gt_vertices.npy'
        colors_path = 'pointcloud/gt_colors.npy'

    elif select_poincloud == 2:
        # load inliers
        # inliers are the points of plane
        vertices_path = 'pointcloud/inlier_vertices.npy'
        colors_path = 'pointcloud/inlier_colors.npy'

    elif select_poincloud == 3:
        # load outliers
        # outliers are the points of objects
        vertices_path = 'pointcloud/outlier_vertices.npy'
        colors_path = 'pointcloud/outlier_colors.npy'

    elif select_poincloud == 4:
        # load inliers 2
        # inliers are the points of plane
        vertices_path = 'pointcloud/inlier_vertices2.npy'
        colors_path = 'pointcloud/inlier_colors2.npy'

    elif select_poincloud == 5:
        # load outliers 2
        # outliers are the points of objects
        vertices_path = 'pointcloud/outlier_vertices2.npy'
        colors_path = 'pointcloud/outlier_colors2.npy'
    



    # load vertices and colors
    vertices = np.load(vertices_path)
    colors = np.load(colors_path)

    print(vertices.shape)
    print(colors.shape)

    # visualize point cloud
    U.visualize_point_cloud(vertices, colors)

main()
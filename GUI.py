import numpy as np
import utility as U

def main():
    # 1 -> my pointcloud
    # 2 -> gt pointcloud
    # 3 -> inliers (ground truth planes)
    # 4 -> outliers (ground truth objects)
    # 5 -> inliers (my planes)
    # 6 -> outliers (my objects)

    select_pointcloud = 6

    if select_pointcloud == 0:
        # pointcloud paths
        vertices_path = 'pointcloud/vertices.npy'
        colors_path = 'pointcloud/colors.npy'

    elif select_pointcloud == 1:
        # gt_pointcloud paths
        vertices_path = 'pointcloud/gt_vertices.npy'
        colors_path = 'pointcloud/gt_colors.npy'

    elif select_pointcloud == 2:
        # load inliers
        # inliers are the points of plane
        vertices_path = 'pointcloud/gt_planes.npy'
        colors_path = 'pointcloud/gt_planes_colors.npy'

    elif select_pointcloud == 3:
        # load outliers
        # outliers are the points of objects
        vertices_path = 'pointcloud/gt_objects.npy'
        colors_path = 'pointcloud/gt_objects_colors.npy'

    elif select_pointcloud == 4:
        # load inliers
        # inliers are the points of plane
        vertices_path = 'pointcloud/my_planes.npy'
        colors_path = 'pointcloud/my_planes_colors.npy'

    elif select_pointcloud == 5:
        # load outliers
        # outliers are the points of objects
        vertices_path = 'pointcloud/my_objects.npy'
        colors_path = 'pointcloud/my_objects_colors.npy'

    elif select_pointcloud == 6:
        vertices_path = 'pointcloud/gt_objects_clustered.npy'
        colors_path = 'pointcloud/gt_objects_colors_clustered.npy'


    # load vertices and colors
    vertices = np.load(vertices_path)
    colors = np.load(colors_path)

    print(vertices.shape)
    print(colors.shape)

    # visualize point cloud
    U.visualize_point_cloud(vertices, colors)

main()
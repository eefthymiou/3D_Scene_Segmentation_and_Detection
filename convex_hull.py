import numpy as np
import open3d as o3d
import utility as U
from scipy.spatial import ConvexHull, convex_hull_plot_2d
import matplotlib.pyplot as plt
plt.style.use('seaborn')

if __name__ == "__main__":
    # load point cloud
    X = np.load("pointcloud/gt_objects_clustered.npy")
    colors = np.load("pointcloud/gt_objects_colors_clustered.npy")

    print("vertices shape:",X.shape)

    # create a convex hull mesh
    hull = ConvexHull(X)

    print(hull.vertices.shape)

    # Get the indices of points in convex hull
    indices = hull.vertices
    # delete these points from the point cloud and color array
    remaining_vertices = np.delete(X, indices, axis=0)
    remaining_colors = np.delete(colors, indices, axis=0)

    fig = plt.figure(figsize = (20,10),facecolor="w") 
    ax = plt.axes(projection="3d") 

    for simplex in hull.simplices:
        ax.plot3D(X[simplex, 0], X[simplex, 1],X[simplex, 2], 's-') 
    
    ax.scatter3D(remaining_vertices[:, 0], remaining_vertices[:, 1], remaining_vertices[:, 2], c=remaining_colors, marker='o')
    
    plt.show()

    
    

    



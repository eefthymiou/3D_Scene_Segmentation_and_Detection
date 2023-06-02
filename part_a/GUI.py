import partA_utility as U
import numpy as np


def main():

    # load vertices and colors
    vertices = np.load('./part_a/vertices.npy')
    colors = np.load('./part_a/colors.npy')

    # visualize point cloud
    U.visualize_point_cloud(vertices, colors)

main()
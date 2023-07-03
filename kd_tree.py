from cmath import inf
import numpy as np
import open3d as o3d
import heapq

class kd_node:

    def __init__(self, index, left_child, right_child):

        self.index = index
        
        self.left_child = left_child
        self.right_child = right_child
    
    def depth_first_search(self):

        indices = depth_first_search(self, indices = [])
        return indices

class kd_tree:

    def __init__(self, points:np.array):

        self.root = self.build_kd_tree(points) 

    def build_kd_tree(self, points:np.array): 

        root = build_kd_tree(points, dim = points.shape[1], indices = np.arange(len(points)), level=0)
        return root

    def get_nodes_of_level(self, level):

        nodes = get_nodes_of_level(self.root, level, nodes = [])
        return nodes

    def find_nearest_neighbor(self, points, id):
        
       _, istar = find_nearest_neighbor(points, id, dim = points.shape[1], level=0, root=self.root, dstar = inf, istar=-1)
       return istar

    def find_points_in_radius(self, points, id, radius):
        indices = find_points_in_radius(points, id, radius, points.shape[1], level=0, root = self.root, indices=[])
        return indices
    
    def find_points_in_sphere(self, points, center, radius):
        indices = find_points_in_sphere(points, center, radius, points.shape[1], level=0, root = self.root, indices=[])
        return indices

    def find_k_nearest_neighbors(self, points, id, K):
        
        heap,_,_ = find_k_nearest_neighbors(points, id, K, points.shape[1], level=0, root = self.root, heap =[], dstar = inf, length = 0)

        indices = []
        while heap:
            _, index = heapq.heappop(heap)
            indices.append(index)

        return indices

def build_kd_tree(points:np.array, dim, indices, level):

    if (len(indices) == 0):
        return
    else:
        # axis = 0,1,2
        axis = level % dim

        # παίρνουμε τα σημεία και τα ταξινομούμε ως προς μιο συντεταγμένη axis (0,1,2 -> x,y,z)
        order = np.argsort(points[indices,axis]) 
        sorted_indices = indices[order]

        median_idx = (len(indices)-1) // 2

        index = sorted_indices[median_idx]

        indices_left = sorted_indices[:median_idx]
        indices_right = sorted_indices[median_idx+1:]

        left_child = build_kd_tree(points, dim, indices = indices_left, level = level+1)
        right_child = build_kd_tree(points, dim, indices = indices_right, level = level+1)

        return kd_node(index = index, left_child = left_child, right_child = right_child)

def get_nodes_of_level(root:kd_node, level, nodes):

    if level==0:
        nodes.append(root)
    else:
        if root.left_child:
            nodes = get_nodes_of_level(root.left_child, level-1, nodes)
        if root.right_child:
            nodes = get_nodes_of_level(root.right_child, level-1, nodes)
    return nodes

def depth_first_search(root:kd_node, indices):
    
    if root.left_child:
        indices = depth_first_search(root.left_child, indices)
    if root.right_child:
        indices = depth_first_search(root.right_child, indices)

    indices.append(root.index)

    return indices

def find_nearest_neighbor(points:np.array, id, dim, level, root, dstar, istar):
    
    axis = level % dim
    d_ = points[id,axis] - points[root.index,axis]

    is_on_left = d_ < 0

    if is_on_left:
        if root.left_child:
            # dstar -> minimum distance
            # istar -> index of node with minimum distance
            dstar, istar = find_nearest_neighbor(points, id, dim, level+1, root.left_child, dstar, istar)
        
        if d_**2 < dstar:
            if root.right_child:
                dstar, istar = find_nearest_neighbor(points, id, dim, level+1, root.right_child, dstar, istar)

    else:
        if root.right_child:
            dstar, istar = find_nearest_neighbor(points, id, dim, level+1, root.right_child, dstar, istar)

        if d_**2 < dstar:
            if root.left_child:
                dstar, istar = find_nearest_neighbor(points, id, dim, level+1, root.left_child, dstar, istar)
          
    if root.index == id:
        pass 
    else:
        d = np.sum(np.square(points[root.index,:]-points[id,:]))

        if d<dstar:
            dstar=d
            istar=root.index

    return dstar, istar

def find_points_in_sphere(points:np.array, center, radius, dim, level, root, indices):
    axis = level % dim
    d_ = center[axis] - points[root.index,axis]

    is_on_left = d_ < 0

    if is_on_left:
        if root.left_child:
            # dstar -> minimum distance
            # istar -> index of node with minimum distance
            indices = find_points_in_sphere(points, center, radius, dim, level+1, root.left_child, indices)
        
        if d_**2 < radius:
            if root.right_child:
                indices = find_points_in_sphere(points, center, radius, dim, level+1, root.right_child, indices)
    else:
        if root.right_child:
            indices = find_points_in_sphere(points, center, radius, dim, level+1, root.right_child, indices)

        if d_**2 < radius:
            if root.left_child:
                indices = find_points_in_sphere(points, center, radius, dim, level+1, root.left_child, indices)
    
    if root.index == id:
        pass
    else:
        d = np.sum(np.square(points[root.index,:]-center))

        if d<radius:
            indices.append(root.index)
    
    return indices

def find_points_in_radius(points:np.array, id, radius, dim, level, root, indices):

    axis = level % dim
    d_ = points[id,axis] - points[root.index,axis]

    is_on_left = d_ < 0

    if is_on_left:
        if root.left_child:
            # dstar -> minimum distance
            # istar -> index of node with minimum distance
            indices = find_points_in_radius(points, id, radius, dim, level+1, root.left_child, indices)
        
        if d_**2 < radius:
            if root.right_child:
                indices = find_points_in_radius(points, id, radius, dim, level+1, root.right_child, indices)

    else:
        if root.right_child:
            indices = find_points_in_radius(points, id, radius, dim, level+1, root.right_child, indices)

        if d_**2 < radius:
            if root.left_child:
                indices = find_points_in_radius(points, id, radius, dim, level+1, root.left_child, indices)
          
    if root.index == id:
        pass 
    else:
        d = np.sum(np.square(points[root.index,:]-points[id,:]))

        if d<radius:
            indices.append(root.index)


    return indices



def find_k_nearest_neighbors(points:np.array, id, K, dim, level, root, heap, dstar, length):
    axis = level % dim
    d_ = points[id,axis] - points[root.index,axis]

    is_on_left = d_ < 0

    if is_on_left:
        if root.left_child:
            # dstar -> minimum distance
            heap, dstar, length = find_k_nearest_neighbors(points, id, K, dim, level+1, root.left_child, heap, dstar, length)
        
        if d_**2 > dstar and length == K:
            pass
        else:
            if root.right_child:
                heap, dstar, length = find_k_nearest_neighbors(points, id, K, dim, level+1, root.right_child, heap, dstar, length)

    else:
        if root.right_child:
            heap, dstar, length = find_k_nearest_neighbors(points, id, K, dim, level+1, root.right_child, heap, dstar, length)

        if d_**2 > dstar and length == K :
            pass
        else:
            if root.left_child:
                heap, dstar, length = find_k_nearest_neighbors(points, id, K, dim, level+1, root.left_child, heap, dstar, length)
          
    if root.index == id:
        pass 
    else:
        d = np.sum(np.square(points[root.index,:]-points[id,:]))

        if length < K:
            heapq.heappush(heap,[-d,root.index])
            length += 1
        else:
            if d<dstar:
                heapq.heapreplace(heap,[-d,root.index])
                dstar = -heap[0][0]
            else:
                dstar = -heap[0][0]

    # dstar -> max distance of min values
    return heap, dstar, length
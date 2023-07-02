import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.visualization.gui import MouseEvent
from open3d.visualization.rendering import Camera
import sys
import os
import numpy as np
import time
import utility as U
import threading




material = rendering.MaterialRecord()
material.shader = "defaultUnlit"
# Set material properties
material.base_color = np.array([0.8, 0.8, 0.8, 1.0])  # Dark diffuse color

class AppWindow:

    def __init__(self, width, height, window_name="Lab"):

        self.w_width = width
        self.w_height = height
        self.first_click = True

        #boilerplate - initialize window & scene
        self.window = gui.Application.instance.create_window(window_name, width, height)
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)

        # basic layout
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)

        #set mouse and key callbacks
        self._scene.set_on_key(self._on_key_pressed)
        # self._scene.set_on_mouse(self._on_mouse_pressed)

        #set up camera
        bounds = self._scene.scene.bounding_box
        center = bounds.get_center()
        self._scene.look_at(center, center + [0, 0, 1], [0, 1, 0])
    
        self.geometries = {}
        self.wireframe_on = False
        self.aabb_on = False
        self.pr_comp_on = False
        self.meshSplited = False
        self.gt_clusters = False
        self.gt_clusters_index = 0
        self.my_clusters = False
        self.my_clusters_index = 0
        self.convex_hull = False
        self.last_click = time.time()

    def _on_layout(self, layout_context):
        
        r = self.window.content_rect
        self._scene.frame = r

    def add_geometry(self, geometry, name):
        self.geometries[name] = geometry

    def remove_geometry(self, name):
        self._scene.scene.remove_geometry(name)
    
    def remove_all_geometries(self):
        for name in self.geometries:
            self.remove_geometry(name)
    
    def prepare_pcd(self, pcd):
        pcd = U.unit_sphere_normalization(pcd)
        pcd_center = U.get_center(pcd)
        pcd = U.translate(pcd, -pcd_center)        
        return pcd
    
    def load_pcd(self,vertices_path,colors_path):
        # load vertices and colors
        vertices = np.load(vertices_path)
        colors = np.load(colors_path)
        return vertices, colors
    
    def add_pcd(self,vertices,colors):
        # create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(vertices)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        # prepare point cloud
        pcd = self.prepare_pcd(pcd)
        # add point cloud to scene
        self._scene.scene.add_geometry("pointcloud", pcd, material)
        self.add_geometry(pcd, "pointcloud")

    def cluster(self, index):
        # remove all geometries
        self.remove_all_geometries()

        index = str(index)

        # import point cloud
        if (self.gt_clusters):
            vertices, colors = self.load_pcd('clusters/gt_clusters/cluster_'+index+'/vertices.npy', 'clusters/gt_clusters/cluster_'+index+'/colors.npy')
        elif (self.my_clusters):
            vertices, colors = self.load_pcd('clusters/my_clusters/cluster_'+index+'/vertices.npy', 'clusters/my_clusters/cluster_'+index+'/colors.npy')
        # add point cloud to scene
        self.add_pcd(vertices, colors)

        vertices = U.unit_sphere_normalization_vertices(vertices)

        # calculate convex hull
        convex_hull = U.chull(vertices)
        
        # print type of convext hull (mesh or point cloud)
        print(type(convex_hull))

        # compute the center of the convex hull
        convex_hull_center = np.mean(vertices,axis=0)
        
        if type(convex_hull) == o3d.cpu.pybind.geometry.TriangleMesh:
            # translate the convex hull to the center of the point cloud
            convex_hull = U.translate_mesh(convex_hull, -convex_hull_center)
        if type(convex_hull) == o3d.cpu.pybind.geometry.LineSet:
            print("LineSet")
            # translate the convex hull to the center of the point cloud
            convex_hull = U.translate_LineSet(convex_hull, -convex_hull_center)
        
        # add convex hull to scene
        # self._scene.scene.add_geometry("convex_hull", convex_hull, material)
        self.add_geometry(convex_hull, "convex_hull")

        # check that vertices and convex hull vertices are the same
        print("vertices shape:", vertices.shape)
    
        return gui.Widget.EventCallbackResult.HANDLED
            
        
    def _on_key_pressed(self, event):
        # if not have passed 0.5 sec since last click return
        if time.time() - self.last_click < 0.5:
            return gui.Widget.EventCallbackResult.HANDLED

        # 1 - load my point cloud
        if event.key == ord('1'):
            # remove all geometries
            self.remove_all_geometries()
            # exit from clusters
            self.gt_clusters = False
            self.my_clusters = False

            # import point cloud
            vertices, colors = self.load_pcd('pointcloud/vertices.npy', 'pointcloud/colors.npy')
            # add point cloud to scene
            self.add_pcd(vertices, colors)

            return gui.Widget.EventCallbackResult.HANDLED

        # 2 - load ground truth point cloud
        if event.key == ord('2'):
            # remove all geometries
            self.remove_all_geometries()
            # exit from clusters
            self.gt_clusters = False
            self.my_clusters = False

            # import point cloud
            vertices, colors = self.load_pcd('pointcloud/gt_vertices.npy', 'pointcloud/gt_colors.npy')
            # add point cloud to scene
            self.add_pcd(vertices, colors)

            return gui.Widget.EventCallbackResult.HANDLED
        
        # 3 - load ground truth planes
        if event.key == ord('3'):
            # remove all geometries
            self.remove_all_geometries()
            # exit from clusters
            self.gt_clusters = False
            self.my_clusters = False

            # import point cloud
            vertices, colors = self.load_pcd('pointcloud/gt_planes.npy', 'pointcloud/gt_planes_colors.npy')
            # add point cloud to scene
            self.add_pcd(vertices, colors)

            return gui.Widget.EventCallbackResult.HANDLED

        # 4 - load ground truth objects
        if event.key == ord('4'):
            # remove all geometries
            self.remove_all_geometries()
            # exit from clusters
            self.gt_clusters = False
            self.my_clusters = False

            # import point cloud
            vertices, colors = self.load_pcd('pointcloud/gt_objects.npy', 'pointcloud/gt_objects_colors.npy')
            # add point cloud to scene
            self.add_pcd(vertices, colors)

            return gui.Widget.EventCallbackResult.HANDLED
        
        # 5 - load my planes
        if event.key == ord('5'):
            # remove all geometries
            self.remove_all_geometries()
            # exit from clusters
            self.gt_clusters = False
            self.my_clusters = False

            # import point cloud
            vertices, colors = self.load_pcd('pointcloud/my_planes.npy', 'pointcloud/my_planes_colors.npy')
            # add point cloud to scene
            self.add_pcd(vertices, colors)

            return gui.Widget.EventCallbackResult.HANDLED
        
        # 6 - load my objects
        if event.key == ord('6'):
            # remove all geometries
            self.remove_all_geometries()
            # exit from clusters
            self.gt_clusters = False
            self.my_clusters = False

            # import point cloud
            vertices, colors = self.load_pcd('pointcloud/my_objects.npy', 'pointcloud/my_objects_colors.npy')
            # add point cloud to scene
            self.add_pcd(vertices, colors)

            return gui.Widget.EventCallbackResult.HANDLED
        
        # 7 - load gt clusters
        if event.key == ord('7'):
            self.my_clusters = False
            self.gt_clusters = True
            self.cluster(self.gt_clusters_index)

        # 8 - load my clusters
        if event.key == ord('8'):
            self.gt_clusters = False
            self.my_clusters = True
            self.cluster(self.my_clusters_index)


        # up key (265) - change cluster 
        if event.key == 265:
            if self.gt_clusters:
                if self.gt_clusters_index == 10:
                    self.gt_clusters_index = 0
                else: self.gt_clusters_index += 1
                self.cluster(self.gt_clusters_index)
            if self.my_clusters:
                if self.my_clusters_index == 12:
                    self.my_clusters_index = 0
                else: self.my_clusters_index += 1
                self.cluster(self.my_clusters_index)

        # down key - chage cluster
        if event.key == 266:
            if self.gt_clusters:
                if self.gt_clusters_index == 0:
                    self.gt_clusters_index = 10
                else: self.gt_clusters_index -= 1
                self.cluster(self.gt_clusters_index)
            elif self.my_clusters:
                if self.my_clusters_index == 0:
                    self.my_clusters_index = 12
                else: self.my_clusters_index -= 1
                self.cluster(self.my_clusters_index)
        

        # ENTER - show convex hull
        if event.key == 10:
            if self.gt_clusters:
                if self.convex_hull == False:
                    # find convex_hull from geometries
                    convex_hull = self.geometries["convex_hull"]
                    # add convex hull to scene
                    self._scene.scene.add_geometry("convex_hull", convex_hull, material)
                    self.convex_hull = True
                else:
                    # remove convex hull from scene
                    self._scene.scene.remove_geometry("convex_hull")
                    self.convex_hull = False
            elif (self.my_clusters):
                if self.convex_hull == False:
                    # find convex_hull from geometries
                    convex_hull = self.geometries["convex_hull"]
                    # add convex hull to scene
                    self._scene.scene.add_geometry("convex_hull", convex_hull, material)
                    self.convex_hull = True
                else:
                    # remove convex hull from scene
                    self._scene.scene.remove_geometry("convex_hull")
                    self.convex_hull = False
    
        print(event.key)
        self.last_click = time.time()
        return gui.Widget.EventCallbackResult.HANDLED
            
def main():
    gui.Application.instance.initialize()

    # initialize GUI
    app = AppWindow(1280, 720)

    gui.Application.instance.run()

if __name__ == "__main__":
    main()

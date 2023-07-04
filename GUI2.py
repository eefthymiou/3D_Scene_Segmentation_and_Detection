from collections.abc import Iterable
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
import random



material = rendering.MaterialRecord()
material.base_color = np.array([0.8, 0.8, 0.8, 1.0])  

mat = rendering.MaterialRecord()
mat.base_color = [
    random.random(),
    random.random(),
    random.random(), 1.0
]


class AppWindow:

    def __init__(self, width, height, window_name="Lab"):

        self.w_width = width
        self.w_height = height
        self.first_click = True

        #boilerplate - initialize window & scene
        self.window = gui.Application.instance.create_window(window_name, width, height)
        self._scene = gui.SceneWidget()
        self._scene.scene = rendering.Open3DScene(self.window.renderer)
        self._scene.scene.scene.enable_sun_light(False)

        # basic layout
        self.window.set_on_layout(self._on_layout)
        self.window.add_child(self._scene)

        #set mouse and key callbacks
        self._scene.set_on_key(self._on_key_pressed)
        # self._scene.set_on_mouse(self._on_mouse_pressed)

        #set up camera
        bounds = self._scene.scene.bounding_box
        center = bounds.get_center()
        self._scene.look_at(center, center + [0, 0, 100], [0, 1, 0])
    
        self.geometries = {}
    
        self.gt_clusters = False
        self.gt_clusters_index = 0
        self.num_of_gt_clusters = U.count_directories('clusters/gt_clusters')
        self.my_clusters = False
        self.my_clusters_index = 0
        self.num_of_my_clusters = U.count_directories('clusters/my_clusters')
        self.convex_hull = False
        self.last_click = time.time()
        self.clusters = False
        self.sphere_added = False
        self.sphere_animation_started = False
        self.sphere_center = [0,10,50]
        self.sphere_radius = 30
        self.length_velocity = 5
        self.sphere_velocity = np.zeros(3)

        self.collision = False
        self.num_of_c_hulls = 0
        self.collision_triangle = None
        self.collision_c_hull = None
       
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



    def clusters_visualization(self,ground_truth=True):
        # self remove all geometries
        self.remove_all_geometries()

        if ground_truth:
            self.sphere_center = [0,10,50]
            self.sphere_radius = 30
            self.length_velocity = 5
            directory = 'clusters/gt_clusters'
            num_of_clusters = self.num_of_gt_clusters
        else:
            self.sphere_center = [-90,-20,300]
            self.sphere_radius = 350
            self.length_velocity = 20
            directory = 'clusters/my_clusters'
            num_of_clusters = self.num_of_my_clusters


        c_hulls = []
        for i in range(num_of_clusters):
            vertices, colors = self.load_pcd(directory+'/cluster_'+str(i)+'/vertices.npy',directory+'/cluster_'+str(i)+'/colors.npy')
            # find the convex hull of the cluster
            convex_hull = U.chull(vertices)
            c_hulls = np.append(c_hulls, convex_hull)

            if i==0:
                total_vertices = vertices
                total_colors = colors
            else:
                total_vertices = np.concatenate((total_vertices, vertices), axis=0)
                total_colors = np.concatenate((total_colors, colors), axis=0)


        # normalize the vertices
        total_vertices = np.asarray(total_vertices)
        distance = np.sqrt(((total_vertices * total_vertices).sum(axis = -1)))
        max_distance = np.max(distance)
        total_vertices /= max_distance
        total_vertices *= 1000
        center = np.mean(total_vertices,axis=0)
        total_vertices -= center

        i=0
        for c_hull in c_hulls:
            try: 
                c_hull_points = np.asarray(c_hull.vertices) 
            except:
                continue
            c_hull_points /= max_distance
            c_hull_points *= 1000
            c_hull_points -= center
            # update c_hull
            c_hull.vertices = o3d.utility.Vector3dVector(c_hull_points)
            # self._scene.scene.add_geometry("convex_hull_"+str(i), c_hull, material)
            self.add_geometry(c_hull, "convex_hull_"+str(i))
            i+=1
        self.num_of_c_hulls = i

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(total_vertices)
        pcd.colors = o3d.utility.Vector3dVector(total_colors)

        # add pcd to scene
        self._scene.scene.add_geometry("pointcloud", pcd, material)
        self.add_geometry(pcd, "pointcloud")

        return gui.Widget.EventCallbackResult.HANDLED

    def add_sphere_to_scene(self):
        sphere = o3d.geometry.LineSet.create_from_triangle_mesh(o3d.geometry.TriangleMesh.create_sphere(np.sqrt(self.sphere_radius)))
        # trnaslate sphere
        sphere.translate(self.sphere_center)

        # add sphere to scene
        self._scene.scene.add_geometry("sphere", sphere, mat)
        self.add_geometry(sphere, "sphere")
        self.sphere_added = True

        # set sphere velocity
        # find the center of the sphere 
        self.sphere_velocity = np.zeros(3) - self.sphere_center
        # normalize velocity
        self.sphere_velocity /= np.linalg.norm(self.sphere_velocity)
        
        # noise in velocity
        self.sphere_velocity[0] += random.uniform(-1.2, 1.2)
        
        # multiply by 5
        self.sphere_velocity *= self.length_velocity
        

        return gui.Widget.EventCallbackResult.HANDLED

    def update_sphere(self):
        # find sphere from geometries
        sphere = self.geometries["sphere"]

        # remove sphere from scene
        self._scene.scene.remove_geometry("sphere")

        # translate sphere
        translation_vec = self.sphere_velocity 
        sphere.translate(translation_vec)
        # update the center of the sphere
        self.sphere_center += translation_vec

        # check for collision
        self.collisions()

        if self.collision == True:
            # find the convex hull from geometries
            convex_hull = self.geometries["convex_hull_" + str(self.collision_c_hull)]
            # add convex_hull to the scene with red color
            c_mat = rendering.MaterialRecord()
            c_mat.base_color = np.array([1, 0, 0, 1.0])
            self._scene.scene.add_geometry("convex_hull_"+str(self.collision_c_hull), convex_hull, c_mat)
            

        # add sphere to scene
        self._scene.scene.add_geometry("sphere", sphere, mat)
        

    def collisions(self):

        for i in range(self.num_of_c_hulls):
            # find the convex hull from geometries
            convex_hull = self.geometries["convex_hull_" + str(i)]

            possible_triangles_indices = []
            index = 0
            for triangle in convex_hull.triangles:
                # calculate the plane of the triangle
                N, d = U.plane_from_points(
                    np.asarray(convex_hull.vertices)[triangle[0]],
                    np.asarray(convex_hull.vertices)[triangle[1]],
                    np.asarray(convex_hull.vertices)[triangle[2]]
                )
                # Find the closest point on the plane to the center of the sphere
                D = (np.dot(N, self.sphere_center) + d) / np.linalg.norm(N)
                if D < np.sqrt(self.sphere_radius):
                    possible_triangles_indices.append(index)
                index += 1  

            # print("possible triangle:",len(possible_triangles))

            for index in possible_triangles_indices:
                triangle = convex_hull.triangles[index]

                # check if any triangle vertices are inside the sphere
                vs = [
                    np.asarray(convex_hull.vertices)[triangle[0]],
                    np.asarray(convex_hull.vertices)[triangle[1]],
                    np.asarray(convex_hull.vertices)[triangle[2]]
                ]
                for v in vs:
                    distance = U.distance(v, self.sphere_center)
                    if distance < np.sqrt(self.sphere_radius):
                        self.collision = True
                        self.collision_triangle = index
                        self.collision_c_hull = i
                        print("Collision detected by vertex")
                        return

                # check if any triangle edges intersect the sphere
                for k in range(3):
                    j = (k + 1) % 3
                    closest_point = U.closest_point_on_line(
                        self.sphere_center,
                        convex_hull.vertices[triangle[k]],
                        convex_hull.vertices[triangle[j]]
                    )
                    if U.distance(closest_point, self.sphere_center) < np.sqrt(self.sphere_radius):
                        self.collision = True
                        self.collision_triangle = index
                        self.collision_c_hull = i
                        print("Collision detected by edge")
                        return

    def sphere_animation(self):
        if self.sphere_animation_started: return gui.Widget.EventCallbackResult.HANDLED 
        while True:
            self.sphere_animation_started = True

            # update the scene
            gui.Application.instance.post_to_main_thread(self.window, self.update_sphere)
            time.sleep(1)

            if self.collision == True: 
                self.sphere_animation_started = False
                self.collision = False
                break
            
            
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

        # G - all ground trouth clusters
        if event.key == 103:
            self.gt_clusters = False
            self.my_clusters = False
            self.clusters_visualization(True)
            self.clusters = True
        
        # M - all my clusters
        if event.key == 109:
            self.gt_clusters = False
            self.my_clusters = False
            self.clusters_visualization(False)
            self.clusters = True

        # S - sphere
        if event.key == 115:
            self.gt_clusters = False
            self.my_clusters = False
            if self.clusters: self.add_sphere_to_scene()

        # a - sphere animation 
        if event.key == 97:
            self.gt_clusters = False
            self.my_clusters = False
            if self.sphere_added: 
                threading.Thread(target=self.sphere_animation).start()


        
        # up key (265) - change cluster 
        if event.key == 265:
            if self.gt_clusters:
                self.gt_clusters_index += 1
                if self.gt_clusters_index == self.num_of_gt_clusters:
                    self.gt_clusters_index = 0
                self.cluster(self.gt_clusters_index)
            if self.my_clusters:
                self.my_clusters_index += 1
                if self.my_clusters_index == self.num_of_my_clusters:
                    self.my_clusters_index = 0
                self.cluster(self.my_clusters_index)

        # down key - change cluster
        if event.key == 266:
            if self.gt_clusters:
                self.gt_clusters_index -= 1
                if self.gt_clusters_index == -1:
                    self.gt_clusters_index = self.num_of_gt_clusters-1 
                self.cluster(self.gt_clusters_index)

            elif self.my_clusters:
                self.my_clusters_index -= 1
                if self.my_clusters_index == -1:
                    self.my_clusters_index = self.num_of_my_clusters-1
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

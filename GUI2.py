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
        self.bounding_box = False

        self.last_click = time.time()
        self.clusters = False
        self.sphere_added = False
        self.sphere_animation_started = False
        self.sphere_center = np.zeros(3)
        self.sphere_radius = 0
        self.length_velocity = 0
        self.sphere_velocity = np.zeros(3)

        self.collision = False
        self.num_of_c_hulls = 0
        self.c_hulls = False
        self.num_of_bounding_boxes = 0
        self.bounding_boxes = False

        self.collision_triangle = None
        self.collision_c_hull = None

        self.collision_with_c_hulls = False
        self.collision_with_bounding_boxes = False

        self.ground_truth = False
       
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
        # compute the center of the cluster
        cluster_center = np.mean(vertices,axis=0)
        
        if type(convex_hull) == o3d.cpu.pybind.geometry.TriangleMesh:
            # translate the convex hull to the center of the point cloud
            convex_hull = U.translate_mesh(convex_hull, -cluster_center)
        if type(convex_hull) == o3d.cpu.pybind.geometry.LineSet:
            print("LineSet")
            # translate the convex hull to the center of the point cloud
            convex_hull = U.translate_LineSet(convex_hull, -cluster_center)
        
        # add convex hull to scene
        # self._scene.scene.add_geometry("convex_hull", convex_hull, material)
        self.add_geometry(convex_hull, "convex_hull")

        # calculate the bounding box
        bounding_box = U.find_AABB(vertices)
        # translate the bounding box to the center of the point cloud
        bounding_box.translate(-cluster_center)
        # add bounding box to scene
        # self._scene.scene.add_geometry("bounding_box", bounding_box, material)
        self.add_geometry(bounding_box, "bounding_box")

        # check that vertices and convex hull vertices are the same
        print("vertices shape:", vertices.shape)
    
        return gui.Widget.EventCallbackResult.HANDLED



    def clusters_visualization(self):
        # self remove all geometries
        self.remove_all_geometries()

        if self.ground_truth:
            self.sphere_center = np.array([0.0,10.0,50.0]) 
            self.sphere_radius = 30
            # self.length_velocity = 5
            directory = 'clusters/gt_clusters'
            num_of_clusters = self.num_of_gt_clusters
        else:
            self.sphere_center = np.array([-90.0,-20.0,200.0])
            self.sphere_radius = 350
            # self.length_velocity = 20
            directory = 'clusters/my_clusters'
            num_of_clusters = self.num_of_my_clusters

        c_hulls = []
        bounding_boxes = []
        for i in range(num_of_clusters):
            vertices, colors = self.load_pcd(directory+'/cluster_'+str(i)+'/vertices.npy',directory+'/cluster_'+str(i)+'/colors.npy')
            # find the convex hull of the cluster
            convex_hull = U.chull(vertices)
            c_hulls = np.append(c_hulls, convex_hull)
            # find the bounding box of the cluster
            bounding_box = U.find_AABB(vertices)
            bounding_boxes = np.append(bounding_boxes, bounding_box)
            if i==0:
                total_vertices = vertices
                total_colors = colors
            else:
                total_vertices = np.concatenate((total_vertices, vertices), axis=0)
                total_colors = np.concatenate((total_colors, colors), axis=0)


        # normalize the vertices
        total_vertices = np.asarray(total_vertices)
        distance = np.sqrt(((total_vertices * total_vertices).sum(axis = -1)))
        total_vertices *= 1000
        max_distance = np.max(distance)
        total_vertices /= max_distance   
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


        i=0
        for bounding_box in bounding_boxes:
            print(i)
            # find vertices/points of the bounding box
            bounding_box.scale(1/max_distance, np.zeros(3))
            bounding_box.scale(1000, np.zeros(3))
            bounding_box.translate(-center)
            # self._scene.scene.add_geometry("bounding_box_"+str(i), bounding_box, material)
            self.add_geometry(bounding_box, "bounding_box_"+str(i))
            i+=1 
        
        # compute the bounding box of the total point cloud
        bounding_box = U.find_AABB(total_vertices)
        # self._scene.scene.add_geometry("bounding_box", bounding_box, material)
        self.add_geometry(bounding_box, "bounding_box_"+str(i))
        i+=1
        
        self.num_of_bounding_boxes = i
        print("num of bounding boxes:", self.num_of_bounding_boxes)
              
        # create point cloud
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

        return gui.Widget.EventCallbackResult.HANDLED

    def update_sphere(self):
        # find sphere from geometries
        sphere = self.geometries["sphere"]

        # remove sphere from scene
        self._scene.scene.remove_geometry("sphere")

        # translate sphere
        translation_vec = self.sphere_velocity 
        sphere.translate(translation_vec)
        # update the center of the sphere (np.array)
        self.sphere_center += translation_vec

        if self.collision_with_c_hulls:
            # check for collision
            self.collisions_c_hulls()

            if self.collision == True:
                # find the convex hull from geometries
                convex_hull = self.geometries["convex_hull_" + str(self.collision_c_hull)]
                # add convex_hull to the scene with red color
                c_mat = rendering.MaterialRecord()
                c_mat.base_color = np.array([1, 0, 0, 1.0])
                self._scene.scene.add_geometry("convex_hull_"+str(self.collision_c_hull), convex_hull, c_mat)
        
        if self.collision_with_bounding_boxes:
            # check for collision
            self.collisions_b_boxes()

            if self.collision == True:
                # find the bounding box from geometries
                bounding_box = self.geometries["bounding_box_" + str(self.collision_bounding_box)]
                self._scene.scene.add_geometry("bounding_box_"+str(self.collision_bounding_box), bounding_box, mat)
        

        # add sphere to scene
        self._scene.scene.add_geometry("sphere", sphere, mat)
    
    def collisions_b_boxes(self):
        for i in range(self.num_of_bounding_boxes-1):
            # find the bounding box from geometries
            bounding_box = self.geometries["bounding_box_" + str(i)]
            # check if the sphere is inside the bounding box
            collision = U.is_sphere_inside_bounding_box(self.sphere_center, np.sqrt(self.sphere_radius), bounding_box)
            if collision:
                self.collision = True
                self.collision_bounding_box = i
                print("Collision detected by bounding box")
                return
        
        # check the big bounding box
        bounding_box = self.geometries["bounding_box_" + str(self.num_of_bounding_boxes-1)]
        # check if the sphere is inside the bounding box
        collision = U.is_sphere_inside_bounding_box(self.sphere_center, np.sqrt(self.sphere_radius), bounding_box)
        if not collision:
            self.collision = True
            self.collision_bounding_box = self.num_of_bounding_boxes-1
            print("Collision detected by bounding box")
            return
        
    def collisions_c_hulls(self):
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
        self.sphere_animation_started = True

        if self.collision_with_bounding_boxes:
            if self.ground_truth:
                noise_x = random.randint(-100,100)
                noise_y = random.randint(0,10)
                self.length_velocity = 3
            else:
                noise_x = random.randint(-300,100)
                noise_y = random.randint(-300,40)
                self.length_velocity = 4

        elif self.collision_with_c_hulls:
            if self.ground_truth:
                noises = [[-100,0], [0,10], [0,-10]]
                noise_x, noise_y = random.choice(noises)
                self.length_velocity = 5
            else:
                noises = [[-300,-100], [-100,-50],[20,-50]]
                noise_x, noise_y = random.choice(noises)
                self.length_velocity = 15

        # set sphere velocity
        # find the center of the sphere 
        self.sphere_velocity = np.zeros(3) - self.sphere_center 
        self.sphere_velocity[0] += noise_x
        self.sphere_velocity[1] += noise_y
        # normalize velocity
        self.sphere_velocity /= np.linalg.norm(self.sphere_velocity)
        # set the length of the velocity
        self.sphere_velocity *= self.length_velocity

        while True:
            # update the scene
            gui.Application.instance.post_to_main_thread(self.window, self.update_sphere)
            if self.collision_with_bounding_boxes: time.sleep(0.05)
            if self.collision_with_c_hulls: time.sleep(1)

            if self.collision == True: 
                self.sphere_animation_started = False
                self.collision = False
                if self.collision_with_bounding_boxes : self.collision_with_bounding_boxes = False
                if self.collision_with_c_hulls : self.collision_with_c_hulls = False
                break
            
        return gui.Widget.EventCallbackResult.HANDLED

    def _on_key_pressed(self, event):
        # if not have passed 0.5 sec since last click return
        if time.time() - self.last_click < 0.5:
            return gui.Widget.EventCallbackResult.HANDLED

        # 1 - load ground truth point cloud
        if event.key == ord('1'):
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
        
        # 2 - load ground truth planes
        if event.key == ord('2'):
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

        # 3 - load ground truth objects
        if event.key == ord('3'):
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
        

        # 4 - load my point cloud
        if event.key == ord('4'):
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
            self.ground_truth = True
            self.clusters_visualization()
            self.clusters = True
        
        # M - all my clusters
        if event.key == 109:
            self.gt_clusters = False
            self.my_clusters = False
            self.ground_truth = False
            self.clusters_visualization()
            self.clusters = True

        # S - sphere
        if event.key == 115:
            self.gt_clusters = False
            self.my_clusters = False
            if self.clusters: self.add_sphere_to_scene()

        # a - sphere animation with c_hulls
        if event.key == 97:
            self.gt_clusters = False
            self.my_clusters = False
            if self.sphere_added: 
                self.collision_with_c_hulls = True
                threading.Thread(target=self.sphere_animation).start()
        
        # q - sphere animaation with bounding boxes
        if event.key == 113:
            self.gt_clusters = False
            self.my_clusters = False
            if self.sphere_added: 
                self.collision_with_bounding_boxes = True
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
        

        # C - show convex hull
        if event.key == 99:
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
            
            if self.num_of_c_hulls > 0:
                if self.c_hulls == False:
                    # add convex hulls to the scene
                    for i in range(self.num_of_c_hulls):
                        convex_hull = self.geometries["convex_hull_"+str(i)]
                        self._scene.scene.add_geometry("convex_hull_"+str(i), convex_hull, material)
                    self.c_hulls = True
                else:
                    # remove convex hulls from the scene
                    for i in range(self.num_of_c_hulls):
                        self._scene.scene.remove_geometry("convex_hull_"+str(i))
                    self.c_hulls = False

        # B - show bounding box
        if event.key == 98:
            if self.gt_clusters:
                if self.bounding_box == False:
                    # find bounding_box from geometries
                    bounding_box = self.geometries["bounding_box"]
                    # add bounding box to scene
                    self._scene.scene.add_geometry("bounding_box", bounding_box, material)
                    self.bounding_box = True
                else:
                    # remove bounding box from scene
                    self._scene.scene.remove_geometry("bounding_box")
                    self.bounding_box = False
            elif (self.my_clusters):
                if self.bounding_box == False:
                    # find bounding_box from geometries
                    bounding_box = self.geometries["bounding_box"]
                    # add bounding box to scene
                    self._scene.scene.add_geometry("bounding_box", bounding_box, material)
                    self.bounding_box = True
                else:
                    # remove bounding box from scene
                    self._scene.scene.remove_geometry("bounding_box")
                    self.bounding_box = False 

            if self.num_of_bounding_boxes > 0:
                if self.bounding_boxes == False:
                    # add bounding boxes to the scene
                    for i in range(self.num_of_bounding_boxes):
                        bounding_box = self.geometries["bounding_box_"+str(i)]
                        self._scene.scene.add_geometry("bounding_box_"+str(i), bounding_box, material)
                    self.bounding_boxes = True
                else:
                    # remove bounding boxes from the scene
                    for i in range(self.num_of_bounding_boxes):
                        self._scene.scene.remove_geometry("bounding_box_"+str(i))
                    self.bounding_boxes = False

        # R - remove all geometries
        if event.key == 114:
            self.remove_all_geometries()
            self.gt_clusters = False
            self.my_clusters = False
            self.sphere_added = False
            self.sphere_animation_started = False
            self.collision_with_c_hulls = False
            self.collision_with_bounding_boxes = False
            self.collision = False
            self.collision_triangle = None
            self.collision_c_hull = None
            self.collision_bounding_box = None
            self.num_of_c_hulls = 0
            self.c_hulls = False
            self.num_of_bounding_boxes = 0
            self.bounding_boxes = False
            self.ground_truth = False
                    
        
        print("key pressed:", event.key)
        self.last_click = time.time()
        return gui.Widget.EventCallbackResult.HANDLED
    
    
        
def main():
    print("3D Scene Segmentation and Detetction")
    print("Press:")
    print("1 -> load ground truth point cloud")
    print("2 -> load ground truth planes")
    print("3 -> load ground truth objects")
    print("4 -> load my point cloud")
    print("5 -> load my planes")
    print("6 -> load my objects")
    print("7 -> load gt clusters")
    print("8 -> load my clusters")
    print("up key -> next cluster")
    print("down key -> previous cluster")
    print("B -> show bounding box of each the cluster that appears in the scene")
    print("C -> show convex hull of each the cluster that appears in the scene")
    print("G -> all ground truth clusters")
    print("M -> all my clusters")
    print("S -> add sphere to scene")
    print("A -> sphere animation with c_hulls")
    print("Q -> sphere animation with bounding boxes")
    print("R -> RESET")
    print("ESC -> exit")

    gui.Application.instance.initialize()

    # initialize GUI
    app = AppWindow(1280, 720)

    gui.Application.instance.run()


if __name__ == "__main__":
    main()

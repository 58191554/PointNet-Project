import torch
import os
import numpy as np
import random
import math

class PointDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir, k=3000):
        self.root_dir = root_dir
        self.items = []
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.k = k
        
        # get
        folders = os.listdir(root_dir)
        idx = -1
        # get different objects
        for item_folder in folders:
            if os.path.isdir(os.path.join(root_dir, item_folder)):
                idx += 1
                # record class
                self.classes.append(item_folder)
                self.class_to_idx[item_folder] = idx
                self.idx_to_class[idx] = item_folder
                
                # get train data
                train_folder = os.path.join(root_dir, item_folder, 'train')
                if os.path.exists(train_folder):
                    train_files = [f for f in os.listdir(train_folder) if f.endswith('.off')]
                    self.items.extend([(os.path.join(train_folder, f), idx) for f in train_files])
                
                # get test data
                test_folder = os.path.join(root_dir, item_folder, 'test')
                if os.path.exists(test_folder):
                    test_files = [f for f in os.listdir(test_folder) if f.endswith('.off')]
                    self.items.extend([(os.path.join(test_folder, f), idx) for f in test_files])

    def __getitem__(self, index):
        # load .off file and return vertices and class label
        file_path, class_idx = self.items[index]
        verts, faces = self.get_off_file(file_path)
        points = np.float32(self.sample(verts, faces, self.k))
        return points, class_idx

    def triangle_area(self, pt1, pt2, pt3):
        # compute the area of the triangle by Heron's formula
        side_a = np.linalg.norm(pt1 - pt2)
        side_b = np.linalg.norm(pt2 - pt3)
        side_c = np.linalg.norm(pt3 - pt1)
        s = 0.5 * (side_a + side_b + side_c)
        return max(s * (s - side_a) * (s - side_b) * (s - side_c), 0)**0.5 
    
    def sample_point(self, pt1, pt2, pt3):
        # barycentric coordinates on a triangle
        s, t = sorted([random.random(), random.random()])
        f = lambda i: s * pt1[i] + (t-s) * pt2[i] + (1-t) * pt3[i]
        return [f(0), f(1), f(2)]

    def sample(self, verts, faces, k):
        # k: number of samples
        # return a kx3 matrix of point clouds
        areas = np.zeros((len(faces)))
        verts = np.array(verts)
        for i in range(len(areas)):
            areas[i] = self.triangle_area(verts[faces[i][0]], verts[faces[i][1]], verts[faces[i][2]])
        # sample k points from each face with weights by their areas
        sampled_faces = (random.choices(faces, weights=areas, k=k))
        pointcloud = np.zeros((k, 3))
        for i in range(len(sampled_faces)):
            pointcloud[i] = self.sample_point(verts[sampled_faces[i][0]],
                                               verts[sampled_faces[i][1]],
                                               verts[sampled_faces[i][2]])
        return pointcloud
    
    def __len__(self):
        return len(self.items)
    
    def get_off_file(self, file_path):
        with open(file_path, 'r') as f:
            verts, faces = self.read_off_file(f)
        return verts, faces
    
    def read_off_file(self, file):
        if 'OFF' != file.readline().strip():
            raise('Not a valid OFF header')
        n_verts, n_faces, __ = tuple([int(s) for s in file.readline().strip().split(' ')])
        verts = [[float(s) for s in file.readline().strip().split(' ')] for i_vert in range(n_verts)]
        faces = [[int(s) for s in file.readline().strip().split(' ')][1:] for i_face in range(n_faces)]
        return verts, faces

dataset = PointDataSet('ModelNet10/')
print(dataset[0])
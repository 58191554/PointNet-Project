import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class PointDataSet(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.items = []
        self.classes = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        
        # get
        folders = os.listdir(root_dir)
        
        # get different objects
        for idx, item_folder in enumerate(folders):
            if not os.path.isdir(os.path.join(root_dir, item_folder)):
                continue
            
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
        points = self.read_off_file(file_path)
        return points, class_idx

    def __len__(self):
        return len(self.items)
    
    def read_off_file(self, file_path):
        with open(file_path, 'r') as f:
            lines = f.readlines()
            num_points = int(lines[1].split()[0])
            points = np.zeros((num_points, 3), dtype=np.float32)
            for i in range(num_points):
                point = lines[i + 2].split()
                points[i] = [float(point[0]), float(point[1]), float(point[2])]
        return points

def plot3d(points):
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  # get xyz coordinate data
  x = points[:, 0]
  y = points[:, 1]
  z = points[:, 2]

  # plot vertices
  ax.scatter(x, y, z, c='b', marker='o', s=1)  # c: 点的颜色，marker: 点的形状，s: 点的大小

  # plot line
  for i in range(points.shape[0]):
    ax.plot([x[i], x[(i + 1) % points.shape[0]]],
        [y[i], y[(i + 1) % points.shape[0]]],
        [z[i], z[(i + 1) % points.shape[0]]], c='r', linewidth=0.5)

  # set axis label
  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')

  plt.show()

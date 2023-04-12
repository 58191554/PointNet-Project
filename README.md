# PointNet-Project
This is a group work project in UC Berkeley cs182. We are working on constructing [this paper: PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation](https://arxiv.org/abs/1612.00593)

PointDataSet API说明文档
类定义
PointDataSet(root_dir)
PointDataSet类是一个继承自torch.utils.data.Dataset的数据集类，用于加载包含点云数据的.off文件。

参数
root_dir (str): 数据集的根目录路径。
方法
init(self, root_dir)
初始化PointDataSet类的实例。

参数
root_dir (str): 数据集的根目录路径。
示例
```python
Copy code
dataset = PointDataSet(root_dir='data/point_clouds')
getitem(self, index)
```
通过索引获取数据集中的一个样本。

参数
index (int): 样本的索引。
返回值
points (numpy.ndarray): 从.off文件中读取的点云数据，形状为(num_points, 3)，其中num_points为点的数量。
class_idx (int): 样本的类别索引。
示例
```python
Copy code
points, class_idx = dataset[0]
len(self)
 ```
获取数据集的样本数量。

返回值
length (int): 数据集的样本数量。
示例
```python
Copy code
length = len(dataset)
read_off_file(self, file_path)
```
从.off文件中读取点云数据。

参数
file_path (str): .off文件的路径。
返回值
points (numpy.ndarray): 从.off文件中读取的点云数据，形状为(num_points, 3)，其中num_points为点的数量。
示例
```python
Copy code
points = dataset.read_off_file('data/point_clouds/item1/train/001.off')
plot3d(points)
```
将点云数据以3D图形的形式绘制出来。

参数
points (numpy.ndarray): 点云数据，形状为(num_points, 3)，其中num_points为点的数量。
示例
```python
Copy code
points = dataset.read_off_file('data/point_clouds/item1/train/001.off')
plot3d(points)
```

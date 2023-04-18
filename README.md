#### PreSampleDataset(torch.utils.data.Dataset):
- 输入root_dir, folder="train"
  - 直接读取了root_dir中提前sample好的torch数据文件(.pt)，所有文件结构和ModelNet的数据集相同，root是/pre_sample/
  - folder参数可以为"train"或者"test",分别读取train或者test的数据
- __getitem__(index)
  - 输出1024x3的点云，数据类型为torch.Tensor.float16()
  - 输出one_hot torch.Tensor 10x1代表种类
- 接入DataLoader
```python
batch_size = 32
shuffle = True
num_workers = 4

dataloader = torch.utils.data.DataLoader(pre_sample_data_train, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
dataloader = torch.utils.data.DataLoader(pre_sample_data_test, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
```
#### pcshow()
- 输入一个地址数据\*pointCloud.T
- 输出点云图片
![](https://github.com/58191554/PointNet-Project/edit/main/img/pic1.png)

<div align="center">
<img src="https://github.com/58191554/PointNet-Project/blob/main/img/T-net_pipeline.drawio%20(1).png"></img>
</div>

给定一个batch的点云信息，希望根据这个batch中所有点的信息计算出一个旋转矩阵来得到cononical 的点云。

首先基层利用一维卷积将输入的每个点的三维坐标升为1024维的高维坐标。之后利用Max pooling 得到高维空间中最大的值。之后将batch个向量拼接成一个向量，输入几层全连接层中。最后将MLP输出的向量分割成batch个$\theta$矩阵。

模型的设计原因：

1. Cov1d或者MLP升维并使用MLP在高维空间中找到每个点最大的维度并Maxpooling是为了得到一个物体在特定角度下的特征向量。
2. 将同一个batch的不同物体在不同角度下的数据拼接并MLP的原因是为了参考一个batch中所有的物体角度，得到旋转矩阵
3. 将旋转矩阵加上Identity matrix是为了在初始情况下，不要出现乘了output矩阵之后点云变变成0的情况，后面的网络若接收全0作为输入，输出毫无意义。





Given a batch of point cloud information, the goal is to calculate a rotation matrix based on the information of all points in the batch in order to obtain canonical point clouds.

The design of the model is as follows:

1. Utilize 1D convolution to increase the three-dimensional coordinates of each input point to a high-dimensional coordinate of 1024 dimensions. Then, use Max pooling to obtain the maximum value in the high-dimensional space.
2. Concatenate the batch of vectors into one vector and input it into several fully connected layers. The output vector from the MLP is then divided into batch-sized $\theta$ matrices.
3. The reasons for the design of the model are as follows:
   - Using Cov1d or MLP to increase the dimension and finding the maximum dimension of each point in the high-dimensional space with MLP and Maxpooling is to obtain the feature vector of an object at a specific angle.
   - Concatenating the data of different objects in the same batch at different angles and using MLP is to refer to the angles of all objects in the batch and obtain the rotation matrix.
   - Adding an identity matrix to the rotation matrix is to avoid the situation where the point cloud becomes all zeros after multiplying with the output matrix in the initial state, as the subsequent network would receive meaningless outputs if it takes an all-zero input.

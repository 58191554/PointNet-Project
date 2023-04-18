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


![](PointNet-Project/img/T-net_pipeline.drawio (1).png)

### PreSampleDataset(torch.utils.data.Dataset):
- 输入root_dir, folder="train"
  - 直接读取了提前sample好的torch数据文件(.pt)，所有文件结构和ModelNet的数据集相同，root是/pre_sample/
- __getitem__(index)
  - 输出1024x3的点云，数据类型为torch.Tensor.float16()
  - 输出one_hot torch.Tensor 10x1代表种类
  

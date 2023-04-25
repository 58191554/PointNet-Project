## Key Concept of HW
### Background
1. Point Cloud Data

   Point clouds are sets of 3D points that represent the shape and structure of objects or scenes in the form of unordered point sets. Due to the unordered property of the point cloud, traditional network can not directly use the point cloud data to train.
2. Point Net
  
   PointNet was introduced by Charles R. Qi et al. in a 2017 paper titled "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation." It is a neural network architecture that directly takes raw point clouds as input and processes them without any pre-processing or feature extraction steps. PointNet is capable of learning meaningful features from point clouds, such as local and global geometric information, and can be used for tasks such as 3D object classification, semantic segmentation, and point cloud generation.
     
### Core Concept of Paper
1. Adding Permutation Invariance

   The idea of extracting a global feature is essential when doing classification and segmentation tasks because the input data cannot reflect the global information. The main challenge with processing point cloud data is that it is unordered. We want the same output features regardless of the order of the input point cloud. To have a permutation-invariant network, we need a symmetric function for unordered input. To handle the above problem, we apply MLPs on the individual points in the input point cloud to expand the dimension of input. And we achieve permutation-invariant by applying max-pooling operations to aggregate information across the points.

2. Joint Alignment Network

   The Joint Alignment Network (T-Net) can eliminate the influence of translation, rotation, and other factors on point cloud classification and segmentation. The input point cloud passes through this network to obtain an affine transformation matrix. Then, this matrix is multiplied by the original point cloud to obtain a new nx3 point cloud. The same method is applied to the feature space to ensure the invariance of features.

3. Concatenating data and global feature during segmentation

   Different from object classification which just needs global feature information, part segmentation needs both global and local feature. To do local and global information aggregation, we use a method similar to U- Net. PointNet aggregate local and global features simply by concentrating the global feature and local feature of each point in the hidden layer of MLP. With this modification, the network can predict each point category relying on global semantics and local geometry.

### How HW engages with the concept
1. Adding Permutation Invariance
   Let students achieve forward pass of MLPs with shared weights and max-pooling across the points.
3. Joint Alignment Network
4. Concatenating data and global feature during segmentation

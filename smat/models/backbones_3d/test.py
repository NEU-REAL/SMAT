import spconv
import torch
features = torch.randn(10, 4）# your features with shape [N, numPlanes]
indices = torch.randn(10, 4)# your indices/coordinates with shape [N, ndim + 1], batch index must be put in indices[:, 0]
spatial_shape = # spatial shape of your sparse tensor, spatial_shape[i] is shape of indices[:, 1 + i].
batch_size = 1 # batch size of your sparse tensor.
x = spconv.SparseConvTensor(features, indices, spatial_shape, batch_size)
x_dense_NCHW = x.dense() # convert sparse tensor to dense NCHW tensor.
print(x.sparity) # helper function to check sparity. 
import torch.nn

# A simple 1D tensor
points = torch.tensor([1.0, 4.0, 2.0, 1.0, 3.0, 5.0])

print("Points: {}".format(points))
print("1st element:{}".format(float(points[0])))
print("Shape of tensor: {}".format(points.shape))

# We change the shape of the tensor
points = torch.tensor([[1.0, 4.0], [2.0, 1.0], [3.0, 5.0]])
print("Points: {}".format(points))
print("Shape of tensor: {}".format(points.shape))
# we analyze the storage of the tensor
print("Storage:{}".format(points.storage()))
print("Size:{}".format(points.size()))
print("Stride:{}".format(points.stride()))
print("Storage offset:{}".format(points.storage_offset()))
# Should raise an exception if no cuda device is available
points_2 = points.to(device='cuda')
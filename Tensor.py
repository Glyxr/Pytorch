
import torch
import numpy as np

# data = [[1,2],[3,4],[4,5]]
# x_data = torch.tensor(data)
# print(x_data)
#
# x_ones = torch.ones_like(x_data)
# print(x_ones)
# x_rand = torch.rand_like(x_data,dtype = torch.float)
# print(x_rand)
#
# shape = (2,3,2)
# rand_tensor = torch.rand((2,3,2))
# ones_tensor = torch.ones(shape)
# zeros_tensor = torch.zeros(shape)
#
# print(rand_tensor)
# print(ones_tensor)
# print(zeros_tensor)
#
# print(x_data.shape)
# print(x_data.dtype)
# print(x_data.device)

#看一下pyton的切片
tensor = torch.ones(4,4)
tensor[:,1] = 2
print(tensor)

# t1 = torch.cat([tensor,tensor],dim = 0)
# print(t1)

# print(tensor*tensor)  #点乘
# print(tensor @ tensor)  #矩阵乘法
# print(tensor.T)    #转置

# print(tensor.add_(5))   #整体加法
# print(tensor.t_())      #转置







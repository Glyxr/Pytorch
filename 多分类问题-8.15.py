# import torch
# from torchvision import transforms
# from torchvision import datasets
# from torch.utils.data import DataLoader
# import torch.nn.functional as F
# import torch.optim as optim
#
# batch_size = 64
# transform = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,),(0.3081,))
# ])
#
# train_dataset = datasets.MNIST(
#     root = '../dataset/mnist/',
#     train = True,
#     download = True,
#     transform = transform
# )
# train_loader = DataLoader(
#     train_dataset,shuffle = True,batch_size = batch_size
# )
# test_dataset = datasets.MNIST(
#     root = '../dataset/mnist',
#     train = False,
#     download = True,
#     transform = transform
# )
# test_loader = DataLoader(
#     test_dataset,shuffle = False,batch_size = batch_size
# )
#
# class Net(torch.nn.Module):
#     def __init__(self):
#         super(Net,self).__init__()
#         self.l1 = torch.nn.Linear(784,512)
#         self.l2 = torch.nn.Linear(512,256)
#         self.l3 = torch.nn.Linear(256,128)
#         self.l4 = torch.nn.Linear(128,64)
#         self.l5 = torch.nn.Linear(64,10)
#
#     def forward(self,x):
#         x = x.view(-1,784)
#         x = F.relu(self.l1(x))
#         x = F.relu(self.l2(x))
#         x = F.relu(self.l3(x))
#         x = F.relu(self.l4(x))
#         return self.l5(x)
#
# model = Net()
#
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(),lr = 0.01,momentum= 0.5)
#
#
import torch
input=torch.arange(0,6)
i = input @ input
print(i)

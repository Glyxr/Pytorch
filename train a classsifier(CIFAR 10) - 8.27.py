# import torchvision
# import torch
# import torchvision.transforms as transforms
# import numpy
# import matplotlib.pyplot as plt
# #准备数据
# transform = transforms.Compose(
#     [transforms.ToTensor(),
#      transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]
# )
# batch_size = 4
# trainset = torchvision.datasets.CIFAR10(root='./data',train = True,download = True,transform = transform)
#
# trainloader = torch.utils.data.DataLoader(trainset,batch_size = batch_size,shuffle = True,num_workers = 0)
#
# testset = torchvision.datasets.CIFAR10(root = './data',train = False,download = True,transform=transform)
#
# testloader = torch.utils.data.DataLoader(testset,batch_size = batch_size,shuffle =  False,num_workers = 0)
#
# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
#
# #展示数据
# # def imshow(img):
# #     img = img/2 +0.5 #unnormalize
# #     npimg = img.numpy()
# #     plt.show(numpy.transpose(npimg,(1,2,0)))
# #     plt.show()
# #
# # #get some random training images
# # dataiter = iter(trainloader)     #trainloader 可迭代对象
# # images ,labels = dataiter.next()
# def imshow(img):
#     img = img / 2 + 0.5     # unnormalize
#     npimg = img.numpy()
#     plt.imshow(numpy.transpose(npimg, (1, 2, 0)))
#     plt.show()
#
#
# # get some random training images
# dataiter = iter(trainloader)
# images, labels = dataiter.next()
#
# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
#
import torch
a = torch.randn((), device='cpu', requires_grad=True)
print(a.item())
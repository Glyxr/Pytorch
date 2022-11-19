import torch
x = torch.ones(5)  #input tensor
y = torch.zeros(3) #exceptd output
w = torch.randn(5,3,requires_grad = True)
b = torch.randn(3,requires_grad= True)

z = torch.matmul(x,w) + b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z,y)

# print(f'Gradient function for z = {z.grad_fn}')
# print(f'Gradient function for loss = {loss.grad_fn}')

loss.backward(retain_graph = True)
# print(w.grad)
# print(b.grad)

print(z.requires_grad)
#disable grad computation
with torch.no_grad():
    z = torch.matmul(x,w)+b
print(z.requires_grad)

# #other way
# z_det = z.detach()
# print(z_det.requires_grad)

#To mark some parameters in your neural network as frozen parameters. This is a very common scenario for finetuning a pretrained network

#To speed up computations when you are only doing forward pass, because computations on tensors that do not track gradients would be more efficient.

#More on computational graphs

inp = torch.eye(5, requires_grad=True)
print(inp)
o = torch.ones_like(inp)
o[1,0] = 0
print(o)
out = (inp+1).pow(2)
out.backward(o, retain_graph=True)
print(f"First call\n{inp.grad}")
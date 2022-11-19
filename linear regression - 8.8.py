import torch

# y = w*x + b

#data
x = torch.tensor([1,2,3])
y = torch.tensor([4,5,6])

#
learning_rate = 1e-3
device = torch.device('cpu')
dtype = torch.float
w = torch.randn((),device = device,dtype = dtype,requires_grad= True)
b = torch.randn((),device = device,dtype = dtype,requires_grad= True)

for epoch in range(20000):
    y_pred = w*x + b
    loss = (y_pred - y).pow(2).sum()

    if epoch % 100 == 99:
        print(loss.item())
    loss.backward()

    #optimize
    with torch.no_grad():
       w -= learning_rate * w.grad
       b -= learning_rate * b.grad

    w.grad = None
    b.grad = None

print(w,b)



import torch
from torch import nn
import matplotlib.pyplot as plt
import csv

filename = "2a.csv"

# data = open(filename, 'r')
# csvreader = csv.reader(data)
# x_train = [float(line[0])
#            for line in csvreader if not line[0].startswith('#')]
# # Reset iterator position to start
# data.seek(0)
# y_train = [float(line[1])
#            for line in csvreader if not line[0].startswith('#')]
# data.close()

x_train = torch.tensor([[0.0], [1.0]])
y_train = torch.tensor([[1.0], [0.0]])
print("x_train")
print(x_train)
print("y_train")
print(y_train)


class NotRegressionModel(nn.Module):
    def __init__(self):
        # Model variables
        # TODO: Adjust weights
        self.W = torch.rand((1, 1), requires_grad=True)
        self.b = torch.rand((1, 1), requires_grad=True)
        # Predictor

    def logits(self, x):
        return x @ self.W + self.b

    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def loss(self, x, y):
        return nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


model = NotRegressionModel()

print(model.W, model.b)
print("---------------")
# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.1)
for epoch in range(10000):
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    # similar to:
    # model.W -= model.W.grad * 0.01
    # model.b -= model.b.grad * 0.01

    optimizer.zero_grad()  # Clear gradients for next step

    # Print model variables and loss
print("W = %s, b = %s, loss = %s" %
      (model.W, model.b, model.loss(x_train, y_train)))

# Visualize result
plt.plot(x_train, y_train, 'o', label='$(x^{(i)},y^{(i)})$')
plt.xlabel('x')
plt.ylabel('y')
x = torch.linspace(torch.min(x_train), torch.max(x_train), 100).reshape(-1, 1)
plt.plot(x, model.f(x).detach(), label='$f(x) = xW+b$')
plt.legend()
plt.show()

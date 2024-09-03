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

x_train = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y_train = torch.tensor([[1.0], [1.0], [1.0], [0.0]])
print("x_train")
print(x_train)
print("y_train")
print(y_train)


class NandRegressionModel(nn.Module):
    def __init__(self):
        # Model variables
        # TODO: Adjust weights
        self.W = torch.rand((2, 1), requires_grad=True)
        self.b = torch.rand((1, 1), requires_grad=True)
        # Predictor

    def logits(self, x):
        return x @ self.W + self.b

    def f(self, x):
        return torch.sigmoid(self.logits(x))

    def loss(self, x, y):
        return nn.functional.binary_cross_entropy_with_logits(self.logits(x), y)


model = NandRegressionModel()

print("weights and bias")
print(model.W, model.b)
print("---------------")
# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.1)
for epoch in range(100000):
    if epoch % 1000 == 0:
        print(epoch)
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
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(x_train[:, 0], x_train[:, 1], y_train[:, 0],
           marker='o', label='$(x^{(i)},y^{(i)})$')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
steps = 10
x = torch.linspace(-0.1, 1.1, steps)
y = torch.linspace(-0.1, 1.1, steps)
grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
xy_grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
z = model.f(xy_grid).detach().reshape(steps, steps)
ax.plot_wireframe(grid_x, grid_y, z)
plt.legend()
plt.show()

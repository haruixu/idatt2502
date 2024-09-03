import torch
from torch import nn
import matplotlib.pyplot as plt


x_train = torch.tensor(
    [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
y_train = torch.tensor([[0.0], [1.0], [1.0], [0.0]], dtype=torch.float32)
print("x_train")
print(x_train)
print("y_train")
print(y_train)


class XorRegressionModel(nn.Module):
    def __init__(self):

        super(XorRegressionModel, self).__init__()
        # Model variables
        self.W1 = nn.Parameter(torch.rand((2, 2), dtype=torch.float32)*2-1)
        self.b1 = nn.Parameter(torch.rand((1, 2), dtype=torch.float32)*2-1)
        self.W2 = nn.Parameter(torch.rand((2, 1), dtype=torch.float32)*2-1)
        self.b2 = nn.Parameter(torch.rand((1, 1), dtype=torch.float32)*2-1)

    def f(self, x):
        # First layer with sigmoid activation
        hidden = torch.sigmoid(x @ self.W1 + self.b1)
        # Second layer with sigmoid activation
        output = torch.sigmoid(hidden @ self.W2 + self.b2)
        return output

    def loss(self, x, y):
        return nn.functional.binary_cross_entropy(self.f(x), y)


model = XorRegressionModel()
print("weights and bias\n", model.W1, model.W2, model.b1, model.b2)

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W1, model.b1, model.W2, model.b2], 0.5)
for epoch in range(20000):
    if epoch % 1000 == 0:
        print(epoch, model.loss(x_train, y_train))
    model.loss(x_train, y_train).backward()  # Compute loss gradients
    optimizer.step()  # Perform optimization by adjusting W and b,
    optimizer.zero_grad()  # Clear gradients for next step


# Visualize result
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(x_train[:, 0], x_train[:, 1], y_train[:, 0],
           marker='o', label='$(x^{(i)},y^{(i)})$')

ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
steps = 10
x = torch.linspace(0, 1, steps)
y = torch.linspace(0, 1, steps)
grid_x, grid_y = torch.meshgrid(x, y)
xy_grid = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
z = model.f(xy_grid).detach().reshape(steps, steps)
ax.plot_surface(grid_x, grid_y, z)
plt.legend()
plt.show()

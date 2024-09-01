import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

filename = "1b.csv"

with open(filename, mode="r")as file:
    csvFile = csv.reader(file)
    x_values = []
    y_values = []

    count = 0

    for lines in csvFile:
        if lines[0].startswith("#"):
            continue
        y_values.append(float(lines[0]))
        x_values.append(float(lines[1]))
        x_values.append(float(lines[2]))

x_train = torch.tensor(x_values).reshape(-1, 2)
y_train = torch.tensor(y_values).reshape(-1, 1)


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0], [0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

        # Predictor
    def f(self, x):
        # print("multiplying: ", x.detach(), " @ ", self.W.detach())
        return x @ self.W + self.b

        # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], lr=0.0001)

size = 10000
for epoch in range(size):
    if epoch % (size // 10) == 0:
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

fig = plt.figure('Linear regression: 3D')

plot1 = fig.add_subplot(111, projection='3d')

x1_train = x_train[:, 0].squeeze()
x2_train = x_train[:, 1].squeeze()

plot1.plot(x1_train, x2_train, y_train[:, 0].squeeze(
), 'o', label='$(x_1^{(i)},x_2^{(i)},y^{(i)})$', color='blue')

grid_size = 50
# determine dimensions of grid by selecting number of points along each dimension
x = torch.linspace(
    torch.min(x_train[:, 0]), torch.max(x_train[:, 0]), grid_size)
print("x", x)
# Create the square matrix
x = x.expand(x.shape[0], -1)
print("x", x)
y = torch.linspace(
    torch.min(x_train[:, 1]), torch.max(x_train[:, 1]), grid_size)
print("y", y)
# Create the square matrix
y = y.expand(x.shape[0], -1)
print("y", y)

# TODO: transpose or note?
y = y.T

# NOTE: x and y grids are 50x50. The combined grid is a 2500x2 grid, each representing a single xy-pair/point, of which there are 2500 of(50x50).
# NOTE: In essence, the combined grid is every x, multiplied by every y
# NOTE: The 2500x2 dimensions allow the matrix to be taken as input in the model f(x) which requires a Mx2 matrix, since W is a 2x1 matrix
combined_grid = torch.stack((x.reshape(-1), y.reshape(-1)), dim=1)
print("----------Rehape(-1)--------")
print(x.reshape(-1))
print(y.reshape(-1))
print("----------combined xy grid-----")
print(combined_grid)

# NOTE: Reshape the results into a [50, 50] grid from [2500, 1] grid in order to plot the results
predictions = model.f(combined_grid).detach().reshape(grid_size, grid_size)
print("---------pred-------")
print(predictions)
print(predictions.size())

plot1.plot_wireframe(x, y, predictions, color='green',
                     label='Model Prediction')

plot1.set_xlabel('vekt')
plot1.set_ylabel('lenngde')
plot1.set_zlabel('alder')
plt.legend()
plt.show()

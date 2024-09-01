import csv
import torch
import matplotlib.pyplot as plt

filename = "1a.csv"

# TODO : Add test data
with open(filename, mode="r")as file:
    csvFile = csv.reader(file)
    x_values = []
    y_values = []
    for lines in csvFile:
        if lines[0].startswith("#"):
            continue
        x_values.append(float(lines[0]))
        y_values.append(float(lines[1]))

x_train = torch.tensor(x_values).reshape(-1, 1)
y_train = torch.tensor(y_values).reshape(-1, 1)


class LinearRegressionModel:
    def __init__(self):
        # Model variables
        self.W = torch.tensor([[0.0]], requires_grad=True)
        self.b = torch.tensor([[0.0]], requires_grad=True)

        # Predictor
    def f(self, x):
        return x @ self.W + self.b

        # Uses Mean Squared Error
    def loss(self, x, y):
        return torch.mean(torch.square(self.f(x) - y))


model = LinearRegressionModel()

print(model.W, model.b)
print("---------------")
# Optimize: adjust W and b to minimize loss using stochastic gradient descent
optimizer = torch.optim.SGD([model.W, model.b], 0.0001)
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
# x = [[1], [6]]]
x = torch.tensor([[torch.min(x_train)], [torch.max(x_train)]])
plt.plot(x, model.f(x).detach(), label='$f(x) = xW+b$')
plt.legend()
plt.show()

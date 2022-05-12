import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from time import perf_counter

# Prep data
X_data_numpy, Y_data_numpy = datasets.make_regression(n_samples=100, n_features=1, noise=40, random_state=3)

# Numpy to Tensor
X_data = torch.from_numpy(X_data_numpy.astype(np.float32))
Y_data = torch.from_numpy(Y_data_numpy.astype(np.float32))

# Reshape Y_data
Y_data = Y_data.view(Y_data.shape[0], 1)

n_samples, n_features = Y_data.shape

# Start ML
process_time_start = perf_counter()

# Create Model
input_size = n_features
output_size = 1
model = nn.Linear(input_size, output_size)

# Define Loss and Optimizer
learning_rate = 0.01

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Training Loop
last_loss = -1
num_epoch = 1000

training_start_time = perf_counter()
for epoch in range(num_epoch):

    # forward and loss
    predicted_data = model(X_data)
    loss = criterion(predicted_data, Y_data)

    # backward pass and update
    loss.backward()
    optimizer.step()

    # check if loss changed
    if last_loss == loss:
        break

    # zero grad and prep next step
    optimizer.zero_grad()
    last_loss = loss.item()

    # print loss and test prediction
    if (epoch + 1) % 10 == 0:
        print(f'{epoch + 1}: Loss: {loss.item():.3f}')

end_time = perf_counter()

# calc training time
training_time = end_time - training_start_time
process_time = end_time - process_time_start

# Plot results
predicted_values = model(X_data).detach().numpy()

plt.plot(X_data_numpy, Y_data_numpy, "ro")
plt.plot(X_data_numpy, predicted_values, "b")
plt.title(f'Training Time: {training_time}\nProcess Time: {process_time}')

plt.show()



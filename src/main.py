
# Librairies
import numpy as np              #To deal with arrays in python
import glob                     #To read filenames in folder

# Our own functions
from read_data import *
from manipulation import *
from SimpleNeuralNetwork import *


# Some variables
root = "data/"
labels = ["RTOE", "LTOE", "RANK", "LANK", "RHEE", "LHEE", "T10"]
learning_rate = 1e-4
iter = 10000
is_printing = True

# We separate the data
(train, test) = split_train_test_files(root)

# We get the data (normalized) and we separate y and X
X_train = get_data_by_labels(labels, train, method="dense")
y_train = X_train[-1:, :].T
X_train = X_train[:-1, :].T
X_test = get_data_by_labels(labels, test, method="dense")
y_test = X_test[-1:, :].T
X_test = X_test[:-1, :].T

# Shuffle the data
perm = np.random.permutation(len(X_train))
X_train = X_train[perm]
y_train = y_train[perm]

# We transform the state in a one hot encoded vector
y = np.zeros((len(y_train), 4))
for i in range (len(y_train)) :
    y[i, (int)(y_train[i])] = 1

if (is_printing) :
    print("##### Infos about the data #####")
    print("X_train : ", X_train.shape)
    print("y_train : ", y_train.shape)
    print("X_test : ", X_test.shape)
    print("y_test : ", y_test.shape)

    print("\nFirst training instance :")
    print(X_train[0, :], " : ", y_train[0, :])

# Hyperparameters
hidden_sizes = [X_train.shape[1], 15, 10, 5]
out_size = 4

# Model
model = SimpleNeuralNetwork(hidden_sizes, out_size)
if (is_printing) : model.display()

# Main algorithm
loss_fn = torch.nn.MSELoss(reduction='sum')                         # Function to compute the loss
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)   # Optimizer used to compute gradient descent

for t in range(iter):

    # Forward pass. This gives us the predictions for the training data.
    # We call it as a function because Module objects override __call__ operator.
    # We give a Tensor and we get a Tensor
    y_pred = model(torch.from_numpy(X_train).float())

    # We compute the loss by comparing the predicted and true values of y.
    # We give tensors and we get a tensor.
    loss = loss_fn(y_pred, torch.Tensor(y))
    if (is_printing) : print(t, loss.item())

    # Setup the gradient to 0 because otherwise they accumulate when we call
    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

if (is_printing) :
    y_pred = model(torch.from_numpy(X_train).float()).tolist()
    for i in range (10) :
        pred = np.argmax(y_pred[i])
        print("Prediction : ", pred, " and original : ", (int)(y_train[i]))

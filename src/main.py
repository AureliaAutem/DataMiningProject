
# Librairies
import numpy as np              #To deal with arrays in python
import glob                     #To read filenames in folder
import matplotlib.pyplot as plt #To plot graphs
import math                     #To use ceil / floor

# Our own functions
from read_data import *
from manipulation import *
from write_data import *
from visualize_data import *
from SimpleNeuralNetwork import *
import matplotlib.pyplot as plt


# reader = btk.btkAcquisitionFileReader()
# reader.SetFilename("CP_GMFCS1_01916_20130128_18.c3d")
# reader.Update()
# acq = reader.GetOutput()
# metadata = acq.GetMetaData()
# point_labels = metadata.FindChild("POINT").value().FindChild("LABELS").value().GetInfo().ToString()
#
#
# acq = sub_x_coord(point_labels, 'CP_GMFCS1_01916_20130128_18.c3d', 'T10')
# write_acq(acq, 'updated.c3d')

# Some variables
root = "data/"
labels = ["RTOE", "LTOE", "RANK", "LANK", "RHEE", "LHEE", "T10"]
learning_rate = 1e-4
iter = 500
is_printing = False
method = "dense"
sub = False

# We separate the data
#(train, validation, test) = split_train_validation_test_files(root)
(blocks, test) = split_cross_validation_test_files(root, 10)


# Hyperparameters to test
middle_sizes = [[2],
                [5],
                [10],
                [15],
                [2, 2],
                [2, 5],
                [5, 5],
                [10, 15],
                [15, 5],
                [15, 10]]

models = []
for i in range (len(middle_sizes)) :

    validation = blocks[i]
    train = []
    for j in range (len(blocks)) :
        if (j != i) :
            train += blocks[j]

    # We get the data (normalized) and we separate y and X
    X_train, y_train = get_data_by_labels(labels, train, method=method, sub=sub)
    X_validation, y_validation = get_data_by_labels(labels, validation, method=method, sub=sub)

    # Out size of the network, 5 for sparse data and 4 for dense data
    out_size = 5

    # Hyperparameters
    hidden_sizes = [X_train.shape[1]] + middle_sizes[i] + [out_size]

    # We transform the state in a one hot encoded vector
    y = np.zeros((len(y_train), 5))
    for k in range (len(y_train)) :
        y[k, (int)(y_train[k])] = 1

    # Model
    models += [SimpleNeuralNetwork(hidden_sizes, out_size, is_printing, iter, learning_rate)]

    print("("+str(i)+") Launch training with "+str(iter)+ " epochs...")
    models[i].train(X_train, y)
    models[i].test(X_validation, y_validation)


div = math.ceil(len(middle_sizes) / 3)
for i in range (len(middle_sizes)) :
    plt.subplot(3, div, i+1)

    time = np.arange(0, len(models[i].loss_history), 1)
    plt.plot(time, models[i].loss_history)

    plt.gca().set_aspect('equal', adjustable='box')
    plt.ylabel('Loss')
    plt.title('Loss history (test accuracy = ' + '{:1.3f}'.format(models[i].test_accuracy) + ')')
    plt.grid(True)

plt.show()

#
# maxi = np.argmax(test_accuracy)
# loss_history = loss_histories[maxi]
# accuracy = test_accuracy[maxi]
# models[i].loss_history = loss_history
# print("Test accuracy : ", accuracy)
# if (sub) :
#     models[i].display_loss_history("Loss history for "+str(iter)+" epochs with "+method+" and centralized data")
# else :
#     models[i].display_loss_history("Loss history for "+str(iter)+" epochs with "+method+" and data")




# # Classify all frames of a video
# index = np.random.randint(len(test))
# X_pred = get_prediction_X(labels, test[index])
# events = get_events_from_dense_model(model, X_pred) # Only for dense representation of data
# # events = get_events_from_sparse_model(model, X_pred) # Only for sparse representation of data
# acq_pred = get_acquisition_from_data(test[index])
# acq_pred = write_event_to_acq(events, acq_pred)
# write_acq_to_file(acq_pred, "testEventWritingSparse.c3d")




# # Displays graphics with classified position of labels in (Y,Z) dimensions
# show_graphic('LTOE', train)
# show_graphic('RTOE', train)
# plt.show()


# Checks if all samples start with X coord in the negatives and ends in the positives
# for sample in train:
#     acq = get_acquisition_from_data(sample)
#     px = acq.GetPoint('T10').GetValues()[(0,-1), 0]
#     print(px)


# Librairies
import numpy as np              #To deal with arrays in python
import glob                     #To read filenames in folder
import matplotlib.pyplot as plt #To plot graphs
import math                     #To use ceil / floor

# Our own functions
from read_data import *
from manipulation import *
from SimpleNeuralNetwork import *
from write_data import *
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


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
iter = 5000
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




# # Displays (Y,Z) coordinates of each point of a file for one label in a graphic
# # for visual interpretation
# x_disp, y_disp = get_2D_disp_data('LTOE', 'CP_GMFCS1_01916_20130128_18.c3d')
# y_disp = y_disp.astype(int) - 1
#
# color_list = ['or', 'ob', 'og']
# color = np.take(color_list, y_disp)
#
# legend_elements = [Line2D([0], [0], marker='o', color='r', label='Only right foot down'),
#                    Line2D([0], [0], marker='o', color='b', label='Only left foot down'),
#                    Line2D([0], [0], marker='o', color='g', label='Both feet down')]
#
# for i in range(len(y_disp)):
#     plt.plot(x_disp[i, 0], x_disp[i, 1], color[i])
#
# plt.legend(handles=legend_elements)
# plt.show()

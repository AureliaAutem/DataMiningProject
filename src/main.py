
# Librairies
import numpy as np              #To deal with arrays in python
import glob                     #To read filenames in folder

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
(train, test) = split_train_test_files(root)
# We get the data (normalized) and we separate y and X
X_train, y_train = get_data_by_labels(labels, train, method=method, sub=sub)
X_test, y_test = get_data_by_labels(labels, test, method=method, sub=sub)

#Out size of the network, 5 for sparse data and 4 for dense data
out_size = 5
# Hyperparameters
hidden_sizes = [X_train.shape[1], 15, 10, out_size]

# We transform the state in a one hot encoded vector
y = np.zeros((len(y_train), 5))
for i in range (len(y_train)) :
    y[i, (int)(y_train[i])] = 1

# Model
model = SimpleNeuralNetwork(hidden_sizes, out_size, is_printing, iter, learning_rate)

print("Launch training with "+str(iter)+ " epochs...")
model.train(X_train, y)

print("Test accuracy : ", model.test(X_test, y_test))

if (sub) :
    model.display_loss_history("Loss history for "+str(iter)+" epochs with "+method+" and centralized data")
else :
    model.display_loss_history("Loss history for "+str(iter)+" epochs with "+method+" and data")




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

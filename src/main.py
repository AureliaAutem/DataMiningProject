
# Librairies
import numpy as np              #To deal with arrays in python
import glob                     #To read filenames in folder

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
iter = 5000
is_printing = True

# We separate the data
(train, test) = split_train_test_files(root)

# We get the data (normalized) and we separate y and X
X_train, y_train = get_data_by_labels(labels, train, method="dense")
X_test, y_test = get_data_by_labels(labels, test, method="dense")



if (is_printing) :
    print("##### Infos about the data #####")
    print("X_train : ", X_train.shape)
    print("y_train : ", y_train.shape)
    print("X_test : ", X_test.shape)
    print("y_test : ", y_test.shape)


input("Launch training with "+str(iter)+ " epochs ?")

# We transform the state in a one hot encoded vector
y = np.zeros((len(y_train), 5))
for i in range (len(y_train)) :
    y[i, (int)(y_train[i])] = 1

# Hyperparameters
hidden_sizes = [X_train.shape[1], 15, 10, 5]
out_size = 5

# Model
model = SimpleNeuralNetwork(hidden_sizes, out_size, is_printing, iter, learning_rate)
if (is_printing) : model.display()

model.train(X_train, y)

if (is_printing) :
    y_pred = model.predict(X_train).tolist()
    for i in range (10) :
        pred = np.argmax(y_pred[i])
        print("Prediction : ", pred, " and original : ", (int)(y_train[i]))

print("Testing accuracy : ", model.test(X_test, y_test))

model.display_loss_history()




# # Classify all frames of a video
# index = np.random.randint(len(test))
# X_pred = get_prediction_X(labels, test[index])
# events = get_events_from_model(model, X_pred) # Only for dense representation of data
# acq_pred = get_acquisition_from_data(test[index])
# acq_pred = write_event_to_acq(events, acq_pred)
# write_acq_to_file(acq_pred, "testEventWriting.c3d")




# # Displays graphics with classified position of labels in (Y,Z) dimensions
# show_graphic('LTOE', test[:10])
# show_graphic('RTOE', test[:10])
# plt.show()

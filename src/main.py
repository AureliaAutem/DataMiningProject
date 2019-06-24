import os
from write_data import *
from read_data import *
from visualize_data import *
import train_NN
import classify
import matplotlib.pyplot as plt

to_execute = 7

# Cross validation for dense data
# Hyperparameters to test
if (to_execute == 0) :
    epochs = 5000
    middle_sizes = [[5],
                    [15],
                    [5, 5],
                    [15, 5],
                    [20, 25],
                    [8, 8, 8]]
    # middle_sizes = [[5],
    #                 [15]]
    train_NN.cross_validation_hidden_layers_sizes(middle_sizes, epochs, "dense")


# Cross validation for dense data
# Hyperparameters to test
if (to_execute == 1) :
    epochs = 5000
    middle_sizes = [[5],
                    [15],
                    [5, 5],
                    [10, 15],
                    [15, 5]]
    # middle_sizes = [[5],
    #                 [15]]
    train_NN.cross_validation_hidden_layers_sizes(middle_sizes, epochs, "sparse")

# Train best model we found and save the results
if (to_execute == 2) :
    labels = ["RTOE", "LTOE", "RANK", "LANK", "RHEE", "LHEE", "T10"]

    # We separate the data
    (train, test) = split_train_test_files("data/")

    # TODO: Train de model

    # Classify all frames of a video
    index = np.random.randint(len(test))
    classify.classify_video(model, labels, test[index], "testEventW2.c3d", "dense")
    pass

# Visualize data for separability
if (to_execute == 3) :
    # Displays graphics with classified position of labels in (Y,Z) dimensions
    (train, test) = split_train_test_files("data/")
    print("Processing the data...")
    show_graphic('LTOE', train)
    show_graphic('RTOE', train)
    show_graphic('T10', train)
    show_graphic('RANK', train)
    plt.show()

# Substract x from a file and visualize it
if (to_execute == 4) :
    sub_x_from_file('CP_GMFCS1_01916_20130128_18.c3d', 'file_with_sub_x.c3d')

if (to_execute == 5) :
    epochs = 5000
    middle_sizes = [[20, 25]]
    for i in range (5) :
        train_NN.cross_validation_hidden_layers_sizes(middle_sizes, epochs, "dense")

if (to_execute == 6) :

    (train, test) = split_train_test_files("data/")

    index = np.random.randint(len(train))
    file = train[index]

    show_labeled_x_speed('LTOE', file)
    show_labeled_x_speed('RTOE', file)
    plt.show()

if (to_execute == 7) :

    (train, test) = split_train_test_files("data/")

    X_train, y_train = get_speed_and_labels(train)

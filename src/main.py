import train_NN
import classify
import matplotlib.pyplot as plt

from write_data import *
from read_data import *
from visualize_data import *
import SimpleNeuralNetwork

to_execute = 0

# Cross validation for dense data
# Hyperparameters to test
if (to_execute == 0) :
    epochs = 5000
    middle_sizes = [[5],
                    [15, 5],
                    [8, 8, 8]]
    train_NN.cross_validation_hidden_layers_sizes(middle_sizes, epochs, "dense")


# Cross validation for dense data
# Hyperparameters to test
if (to_execute == 1) :
    epochs = 5000
    middle_sizes = [[5],
                    [15, 5],
                    [8, 8, 8]]
    train_NN.cross_validation_hidden_layers_sizes(middle_sizes, epochs, "sparse")

# Train best model we found and save the results
if (to_execute == 2) :
    labels = ["RTOE", "LTOE", "RANK", "LANK", "RHEE", "LHEE", "T10"]

    # We separate the data
    (train, test) = split_train_test_files("data/")
    X_train, y_train = get_data_by_labels(labels, train, method="dense", sub=True)

    # TODO: Train de model
    model = SimpleNeuralNetwork.SimpleNeuralNetwork([X_train.shape[1], 15, 5, 3], 3, False, 5000, 1e-4, True)
    y = np.zeros((len(y_train), 3))
    for k in range (len(y_train)) :
        y[k, (int)(y_train[k])] = 1
    model.train(X_train, y)

    # Classify all frames of a video
    index = np.random.randint(len(test))
    classify.classify_video(model, labels, test[index], "testEventW2.c3d", "dense")

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

# Execute several times train with the same parameters to see if the accuracy varies
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

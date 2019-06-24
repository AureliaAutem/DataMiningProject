# Librairies
import numpy as np              #To deal with arrays in python
import glob                     #To read filenames in folder
import matplotlib.pyplot as plt #To plot graphs
import math                     #To use ceil / floor

# Our own functions
from read_data import *
from write_data import *
from visualize_data import *
from SimpleNeuralNetwork import *
import matplotlib.pyplot as plt


def cross_validation_hidden_layers_sizes(middle_sizes, epochs, method) :
    # Some variables
    root = "data/"
    labels = ["RTOE", "LTOE", "RANK", "LANK", "RHEE", "LHEE", "T10"]
    learning_rate = 1e-4
    iter = epochs
    is_printing = False

    if (method == "sparse") :
        out_size = 5
    elif (method == "dense") :
        out_size = 3

    # We separate the data
    #(train, validation, test) = split_train_validation_test_files(root)
    (blocks, test) = split_cross_validation_test_files(root, 10)

    middle_sizes = middle_sizes + middle_sizes
    models = []
    for i in range (len(middle_sizes)) :

        validation_accuracy = []
        for l in range (len(blocks)) :
            validation = blocks[l]
            train = []
            for j in range (len(blocks)) :
                if (j != l) :
                    train += blocks[j]

            # We get the data (normalized) and we separate y and X
            sub = (i < (int)(len(middle_sizes)/2))
            X_train, y_train = get_data_by_labels(labels, train, method=method, sub=sub)
            X_validation, y_validation = get_data_by_labels(labels, validation, method=method, sub=sub)

            # Hyperparameters
            hidden_sizes = [X_train.shape[1]] + middle_sizes[i] + [out_size]

            # We transform the state in a one hot encoded vector
            y = np.zeros((len(y_train), out_size))
            for k in range (len(y_train)) :
                y[k, (int)(y_train[k])] = 1

            # Model
            models += [SimpleNeuralNetwork(hidden_sizes, out_size, is_printing, iter, learning_rate, sub)]

            print("\n("+str(i)+") Launch training with "+str(iter)+ " epochs...")
            models[i].train(X_train, y, register_loss=(l==0))
            validation_accuracy += [models[i].test(X_validation, y_validation)]

        models[i].validation_accuracy = sum(validation_accuracy)/len(validation_accuracy)
        print(models[i], "\nTest accuracy = " + '{:1.3f}'.format(models[i].validation_accuracy))


    div = math.ceil(len(middle_sizes) / 3)
    max_arg = 0
    for i in range (len(middle_sizes)) :
        if (models[max_arg].validation_accuracy < models[i].validation_accuracy) :
            max_arg = i

        plt.subplot(3, div, i+1)

        time = np.arange(0, len(models[i].loss_history), 1)
        plt.plot(time, models[i].loss_history)

        plt.gca().set_aspect('equal', adjustable='box')
        plt.ylabel('Loss')
        plt.title('Loss history (test accuracy = ' + '{:1.3f}'.format(models[i].validation_accuracy) + ')')
        plt.grid(True)

    print("The best model found is : ")
    print(models[max_arg])

    X_test, y_test = get_data_by_labels(labels, test, method=method, sub=models[max_arg].sub)
    print("\nTest accuracy for the best model : ", models[max_arg].test(X_test, y_test))

    plt.show()

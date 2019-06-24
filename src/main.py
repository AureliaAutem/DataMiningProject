import os
import write_data
import train_NN

to_execute = 0

# Cross validation for dense data
# Hyperparameters to test
if (to_execute == 0) :
    epochs = 5000
    middle_sizes = [[5],
                    [15],
                    [5, 5],
                    [10, 15],
                    [15, 5]]
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
    pass

# Visualize data for separability
if (to_execute == 3) :
    pass

# Substract x from a file and visualize it
if (to_execute == 4) :
    write_data.sub_x_from_file('CP_GMFCS1_01916_20130128_18.c3d', 'file_with_sub_x.c3d')

#Librairies
from btk import btk             #To read the data in .c3d files
import numpy as np
import glob


def split_train_test_files(root) :
    """Split files in 2/3 for the training and 1/3 for the tests"""
    folders = [f for f in glob.glob(root+"**/")]

    train = []
    test = []
    for folder in folders :
        filenames = glob.glob(folder+"*.c3d")
        np.random.shuffle(filenames)

        sep = round(len(filenames)/3*2)
        train += filenames[:sep]
        test += filenames[sep:]

    return (train, test)

def get_acquisition_from_data(file) :
    """Get acquisition object to manipulate the data with BTK"""
    # We create a new acquisition object
    reader = btk.btkAcquisitionFileReader() # build a btk reader object
    reader.SetFilename(file) # set a filename to the reader
    reader.Update()
    acq = reader.GetOutput() # acq is the btk aquisition object
    return acq

def scale(X, x_min, x_max):
    """Function to scale each column of a matrix"""
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

def get_data_by_labels(labels, filenames, method="sparse") :
    """Get the data according to the filenames and labels given in parameters"""
    if (method == "sparse") :
        X_train = get_sparse_data_from_file(labels, filenames[0])
        for i in range (1, len(filenames)) :
            res = get_sparse_data_from_file(labels, filenames[i])
            X_train = np.concatenate((X_train, res), axis=1)
    elif (method == "dense") :
        X_train = get_dense_data_from_file(labels, filenames[0])
        for i in range (1, len(filenames)) :
            res = get_dense_data_from_file(labels, filenames[i])
            X_train = np.concatenate((X_train, res), axis=1)
    else :
        print("'" + method + "'" + ' is not a correct method name for the function get_data_by_labels(). Accepted values asre:\n\nsparse\ndense')
        exit(-1)

    return X_train

def get_sparse_data_from_file(labels, filename) :
    # We create an acquisition to manipulate the file c3d
    acq = get_acquisition_from_data(filename)

    # We extract the frames where there are events
    n_events = acq.GetEventNumber() # Number of events

    event_frames = []
    event_labels = []
    event_contexts = []
    for event in [acq.GetEvent(event) for event in range(n_events)] :
        event_frames += [event.GetFrame()]
        event_labels += [event.GetLabel()]
        event_contexts += [event.GetContext()]

    perm = np.argsort(event_frames)
    event_frames = np.take(event_frames, perm)

    event_labels = np.take(event_labels, perm)
    event_labels = (np.array(event_labels) == 'Foot_Strike_GS')

    event_contexts = np.take(event_contexts, perm)
    event_contexts = (np.array(event_contexts) == 'Left')

    first_frame = acq.GetFirstFrame()
    start_frame = event_frames[0]-first_frame
    end_frame = event_frames[-1]-first_frame
    vector = define_sparse_labels(event_frames, event_labels, event_contexts, start_frame, end_frame)

    #print("Number of events : ", n_events)
    #print("Frames where there are events : ", event_frames)

    # We get the data for the selected labels
    # X of shape : (2*nb_labels+2) x (n)
    # 2*nb_labels : all coordinates y and z for each label
    # +2 : predictions
    # n : number of frames
    X = acq.GetPoint(labels[0]).GetValues()[event_frames, 0:3]
    for i in range (1, len(labels)):
        res = acq.GetPoint(labels[i]).GetValues()[event_frames, 0:3]
        X = np.concatenate((X, res), axis=1)

    # X = np.column_stack((X, event_labels.T))
    # X = np.column_stack((X, event_contexts.T))
    X = np.column_stack((X, vector))
    X = scale(X, -1, 1)
    for i in range (10) :
        print(X[i])
    X = X.T

    #print("Matrix X :\n", X)

    return X


def get_dense_data_from_file(labels, filename) :
    # We create an acquisition to manipulate the file c3d
    acq = get_acquisition_from_data(filename)

    # We extract the frames where there are events
    n_events = acq.GetEventNumber() # Number of events
    if (n_events == 0) :
        print("There is no event !")
        exit(-1)

    event_frames = []
    event_labels = []
    event_contexts = []
    for event in [acq.GetEvent(event) for event in range(n_events)] :
        event_frames += [event.GetFrame()]
        event_labels += [event.GetLabel()]
        event_contexts += [event.GetContext()]

    perm = np.argsort(event_frames)
    event_frames = np.take(event_frames, perm)

    event_labels = np.take(event_labels, perm)
    event_labels = (np.array(event_labels) == 'Foot_Strike_GS')

    event_contexts = np.take(event_contexts, perm)
    event_contexts = (np.array(event_contexts) == 'Left')

    first_frame = acq.GetFirstFrame()
    start_frame = event_frames[0]-first_frame
    end_frame = event_frames[-1]-first_frame
    vector = define_dense_labels(event_frames, event_labels, event_contexts, start_frame, end_frame)


    # We get the data for the selected labels
    # X of shape : (2*nb_labels+2) x (n)
    # 2*nb_labels : all coordinates y and z for each label
    # +2 : predictions
    # n : number of frames
    X = acq.GetPoint(labels[0]).GetValues()[start_frame:end_frame, 0:3]
    for i in range (1, len(labels)):
        res = acq.GetPoint(labels[i]).GetValues()[start_frame:end_frame, 0:3]
        X = np.concatenate((X, res), axis=1)
    X = scale(X, -1, 1)

    X = np.column_stack((X, vector)).T
    return X


def define_dense_labels(event_frames, event_labels, event_contexts, start_frame, end_frame) :
    """Function which define to which label an instance belong
        0 : both feet are off
        1 : right foot is down
        2 : left foot is down
        3 : both feet are down
    """
    left = 0; right = 0;
    vector = np.zeros(end_frame - start_frame)

    for i in range(len(event_frames)) :
        index = event_frames[i] - event_frames[0]
        if (event_contexts[i]) :
            if (event_labels[i]) :                  #Left strike
                left = 2
            else :                                  #Left off
                left = 0
        else :
            if (event_labels[i]) :                  #Right strike
                right = 1
            else :                                  #Right off
                right = 0

        if (i == len(event_frames)-1) :
            vector[index : ] = left + right
        else :
            next_index = event_frames[i+1] - event_frames[0]
            vector[index : next_index] = left + right

    return vector;



# def define_sparse_labels(event_frames, event_labels, event_contexts, start_frame, end_frame) :
#     vector = np.zeros(end_frame - start_frame)
#     pad = 2
#
#     for i in range(len(event_frames)) :
#         #print(event_frames[i], ", Foot strike : ", event_labels[i], ", Left :", event_contexts[i])
#         index = event_frames[i] - event_frames[0]
#         if (event_contexts[i]) :
#             if (event_labels[i]) :                  #Left strike
#                 vector[index] = 1
#                 vector = padding(vector, index, 1, pad)
#             else :                                  #Left off
#                 vector[index] = 2
#                 vector = padding(vector, index, 2, pad)
#         else :
#             if (event_labels[i]) :                  #Right strike
#                 vector[index] = 3
#                 vector = padding(vector, index, 3, pad)
#             else :                                  #Right off
#                 vector[index] = 4
#                 vector = padding(vector, index, 4, pad)
#
#     return vector
#
# def padding(vector, index, val, pad) :
#     if (index - pad < 0) :
#         vector[0:index] = val
#     else :
#         vector[index-pad:index] = val
#
#     if (index + pad >= len(vector)) :
#         vector[index:] = val
#     else :
#         vector[index:index+pad] = val
#
#     return vector

def get_2D_disp_data(label, filename) :
    """Can probably be deleted because it just returns the X values and the Y
    labels separatly"""
    # We create an acquisition to manipulate the file c3d
    acq = get_acquisition_from_data(filename)
    x = acq.GetPoint(label).GetValues()[:, 1:3]


    # We extract the frames where there are events
    n_events = acq.GetEventNumber() # Number of events
    if (n_events == 0) :
        print("There is no event !")
        exit(-1)

    event_frames = []
    event_labels = []
    event_contexts = []
    for event in [acq.GetEvent(event) for event in range(n_events)] :
        event_frames += [event.GetFrame()]
        event_labels += [event.GetLabel()]
        event_contexts += [event.GetContext()]

    perm = np.argsort(event_frames)
    event_frames = np.take(event_frames, perm)

    event_labels = np.take(event_labels, perm)
    event_labels = (np.array(event_labels) == 'Foot_Strike_GS')

    event_contexts = np.take(event_contexts, perm)
    event_contexts = (np.array(event_contexts) == 'Left')

    first_frame = acq.GetFirstFrame()
    start_frame = event_frames[0]-first_frame
    end_frame = event_frames[-1]-first_frame
    vector = define_dense_labels(event_frames, event_labels, event_contexts, start_frame, end_frame)

    return x, vector

#Librairies
from btk import btk             #To read the data in .c3d files
import numpy as np
import glob
import random

##### Basic functions for BTK #####
def get_acquisition_from_data(file) :
    """Get acquisition object to manipulate the data with BTK"""
    # We create a new acquisition object
    reader = btk.btkAcquisitionFileReader() # build a btk reader object
    reader.SetFilename(file) # set a filename to the reader
    reader.Update()
    acq = reader.GetOutput() # acq is the btk aquisition object
    return acq

##### Functions to get filenames for data folder #####
def split_train_test_files(root) :
    """Get all c3d files in subfolders of root and split these files in
    3 groups : train (2/3) and test (1/3)
    Inputs :    - root : where are located the subfolders containing the data"""
    # We get the subfolders
    folders = [f for f in glob.glob(root+"**/")]

    train = []
    test = []
    for folder in folders :
        # We get the files and shuffle them
        filenames = glob.glob(folder+"*.c3d")
        np.random.shuffle(filenames)

        # We split the data
        sep = round(len(filenames)/3*2)
        train += filenames[:sep]
        test += filenames[sep:]

    return (train, test)

def split_train_validation_test_files(root) :
    """Get all c3d files in subfolders of root and split these files in
    3 groups : train (1/2), validation (1/4) and test (1/4)
    Inputs :    - root : where are located the subfolders containing the data"""
    # We get the subfolders
    folders = [f for f in glob.glob(root+"**/")]

    train = []
    validation = []
    test = []
    for folder in folders :
        # We get the files and shuffle them
        filenames = glob.glob(folder+"*.c3d")
        np.random.shuffle(filenames)

        # We split the data
        sep = round(len(filenames)/4)
        train += filenames[:2*sep]
        validation += filenames[2*sep:3*sep]
        test += filenames[3*sep:]

    return (train, validation, test)


##### Functions to process the data #####
def get_data_by_labels(labels, filenames, method="sparse", sub=False) :
    """Get the data according to the filenames and labels given in parameters
    Inputs :    - labels : list of points we want to extract in the c3d files
                - filenames : relative path to the data
                - method : describes how we process the data, it can be :
                    - sparse : we extract each frame labeled with an event and
                                we add the same number of frames with no event
                    - dense : we extract all frames from the first event to the
                                last and we label them according to the feet
                                position
    Outputs :   - tuple (X, y)
                    - X : instances of size (N, nb_labels*3)
                    - y : classes of size (N, 1)
    where N is the number of frames and nb_labels is the size of the list
    'labels' that we multiply by 3 because there are 3 coordinates (x, y, z)
    describing each label
    The data is normalized between -1 and 1"""

    acqs = []
    for filename in filenames :
        if (sub) :
            acqs += [sub_x_coord(labels, filename, "T10")]
        else :
            acqs += [get_acquisition_from_data(filename)]


    if (method == "sparse") :
        data = get_sparse_data_from_file(labels, acqs[0])
        for i in range (1, len(acqs)) :
            res = get_sparse_data_from_file(labels, acqs[i])
            data = np.concatenate((data, res), axis=1)
    elif (method == "dense") :
        data = get_dense_data_from_file(labels, acqs[0])
        for i in range (1, len(acqs)) :
            res = get_dense_data_from_file(labels, acqs[i])
            data = np.concatenate((data, res), axis=1)
    else :
        print("'" + method + "'" + ' is not a correct method name for the function get_data_by_labels(). Accepted values asre:\n\nsparse\ndense')
        exit(-1)

    return (data[:-1, :].T, data[-1:, :].T)

def get_sparse_data_from_file(labels, acq) :
    """Extract instances and classes from one particular file in 'sparse' method.
    We extract each frame labeled with an event and we add the same number of
    frames with no event that we choose randomly.
    Then we have 5 differents classes :
    0 : no event
    1 : left foot down
    2 : left foot off
    3 : right foot down
    4 : right foot off
    Inputs :    - labels : list of points we want to extract in the c3d file
                - filename : relative path of the file
    Outputs :   - X : data of size (nb_labels+1, nb_frames)
    where nb_labels is the size of the 'labels' list, we add because we append
    the class and nb_frames is the number of frames we will extract.
    """
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

    vector = define_sparse_labels(event_frames, event_labels, event_contexts)
    vector = np.append(vector, np.zeros(len(event_frames)))

    # We get the data for the selected labels
    frames = define_sparse_no_event_frames(event_frames, event_labels, event_contexts, start_frame, end_frame)
    X = acq.GetPoint(labels[0]).GetValues()[frames, 0:3]
    for i in range (1, len(labels)):
        res = acq.GetPoint(labels[i]).GetValues()[frames, 0:3]
        X = np.concatenate((X, res), axis=1)

    #We normalize each column (each coordinate)
    X = scale(X, -1, 1)
    X = np.column_stack((X, vector))
    X = X.T

    return X


def get_dense_data_from_file(labels, acq) :
    """Extract instances and classes from one particular file in 'dense' method.
    we extract all frames from the first event to the last and we label them
    according to the feet position.
    Then we have 4 differents classes :
    0 : both feet are off
    1 : right foot is down
    2 : left foot is down
    3 : both feet are down
    Inputs :    - labels : list of points we want to extract in the c3d file
                - filename : relative path of the file
    Outputs :   - X : data of size (nb_labels+1, nb_frames)
    where nb_labels is the size of the 'labels' list, we add because we append
    the class and nb_frames is the number of frames we will extract.
    """
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
    X = acq.GetPoint(labels[0]).GetValues()[start_frame:end_frame+1, 0:3]
    for i in range (1, len(labels)):
        res = acq.GetPoint(labels[i]).GetValues()[start_frame:end_frame+1, 0:3]
        X = np.concatenate((X, res), axis=1)
    X = scale(X, -1, 1)
    X = np.column_stack((X, vector)).T
    return X

def sub_x_coord(labels, file, to_sub):
    """Function which substract the value of 'to_sub' to all the points in
    labels from the file 'file'
    Inputs :    - labels : points
                - file : relative path to the file
                - to_sub : point we want to substract
    Outputs :   - acq : acquisition with the modified points
    """
    acq = get_acquisition_from_data(file)

    sub_val = acq.GetPoint(to_sub).GetValues()[:, 0]

    for label in labels :
        label = label.strip()
        # print('Working with [' + label + ']')

        updatedX = acq.GetPoint(label).GetValues()[:, 0]
        updatedX = updatedX - sub_val

        for i in range(len(updatedX)):
            acq.GetPoint(label).SetValue(i, 0, updatedX[i])

        # print('Updated ' + label)
        # print(acq.GetPoint(label).GetValues()[0:10, :])

    return acq



##### Manipulation functions to normalize the data and define the labels #####
def scale(X, x_min, x_max):
    """Function to normalize each column of a matrix
    In our case, we'll normalize between -1 and 1
    Inputs :    - X : data we want to normalize
                - x_min, x_max : bounds for the normalization
    Outputs :   - array with the same size than X"""
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

def define_dense_labels(event_frames, event_labels, event_contexts, start_frame, end_frame) :
    """Function which define to which label each frame belong
        0 : both feet are off
        1 : right foot is down
        2 : left foot is down
        3 : both feet are down
    Inputs :    - event_frames, event_labels, event_contexts : from BTK, to get
                    and classify events
                - start_frame, end_frame : first and last frame with events
    Outputs :   - vector of size (nb_frames, 1)
    """
    left = 0; right = 0;
    vector = np.zeros(end_frame - start_frame + 1)

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



def define_sparse_labels(event_frames, event_labels, event_contexts) :
    """Function which define to which label each frame belong
        0 : no event
        1 : left foot down
        2 : left foot off
        3 : right foot down
        4 : right foot off
    Inputs :    - event_frames, event_labels, event_contexts : from BTK, to get
                    and classify events
                - start_frame, end_frame : first and last frame with events
    Outputs :   - vector of size (nb_frames, 1)
    """
    vector = np.zeros(len(event_frames))

    for i in range(len(event_frames)) :
        if (event_contexts[i]) :
            if (event_labels[i]) :                  #Left strike
                vector[i] = 1
            else :                                  #Left off
                vector[i] = 2
        else :
            if (event_labels[i]) :                  #Right strike
                vector[i] = 3
            else :                                  #Right off
                vector[i] = 4

    return vector

def define_sparse_no_event_frames(event_frames, event_labels, event_contexts, start_frame, end_frame) :
    """Function which choose randomly frames with no event
    Inputs :    - event_frames, event_labels, event_contexts : from BTK, to get
                    and classify events
                - start_frame, end_frame : first and last frame with events
    Outputs :   - vector of size (nb_frames, 1)
    where nb_frames is the length of event_frames
    """
    frames = event_frames.tolist()

    for i in range (len(event_frames)) :
        frame = random.randint(start_frame, end_frame)
        while (frame in frames) :
            frame = random.randint(start_frame, end_frame)

        frames += [frame]

    return frames

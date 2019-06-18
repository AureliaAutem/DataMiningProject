#Librairies
from btk import btk             #To read the data in .c3d files
import numpy as np
import glob


def split_train_test_files(root) :
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
    # We create a new acquisition object
    reader = btk.btkAcquisitionFileReader() # build a btk reader object
    reader.SetFilename(file) # set a filename to the reader
    reader.Update()
    acq = reader.GetOutput() # acq is the btk aquisition object
    return acq


def get_data_by_labels(labels, filenames, method="sparse") :
    X_train = get_data_from_file(labels, filenames[0])
    if (method == "sparse") :
        for i in range (1, len(filenames)) :
            res = get_sparse_data_from_file(labels, filenames[i])
            X_train = np.concatenate((X_train, res), axis=1)
    else :
        for i in range (1, len(filenames)) :
            res = get_dense_data_from_file(labels, filenames[i])
            X_train = np.concatenate((X_train, res), axis=1)

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

    #print("Number of events : ", n_events)
    #print("Frames where there are events : ", event_frames)

    # We get the data for the selected labels
    # X of shape : (2*nb_labels+2) x (n)
    # 2*nb_labels : all coordinates y and z for each label
    # +2 : predictions
    # n : number of frames
    X = acq.GetPoint(labels[0]).GetValues()[event_frames, 1:3]
    for i in range (1, len(labels)):
        res = acq.GetPoint(labels[i]).GetValues()[event_frames, 1:3]
        X = np.concatenate((X, res), axis=1)

    X = np.column_stack((X, event_labels.T))
    X = np.column_stack((X, event_contexts.T))
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

    #print("Number of events : ", n_events)
    #print("Frames where there are events : ", event_frames)

    # We get the data for the selected labels
    # X of shape : (2*nb_labels+2) x (n)
    # 2*nb_labels : all coordinates y and z for each label
    # +2 : predictions
    # n : number of frames
    X = acq.GetPoint(labels[0]).GetValues()[start_frame:end_frame+1, 1:3]
    for i in range (1, len(labels)):
        res = acq.GetPoint(labels[i]).GetValues()[start_frame:end_frame+1, 1:3]
        X = np.concatenate((X, res), axis=1)

    X = np.column_stack((X, event_labels.T))
    X = np.column_stack((X, event_contexts.T))
    X = X.T

    #print("Matrix X :\n", X)

    return X

def interpolate_predictions(event_frames, event_labels, event_contexts, start_frame, end_frame) :
    left = 0; right = 0;
    left_vector = np.zeros(end_frame - start_frame + 1)
    right_vector = np.zeros(end_frame - start_frame + 1)

    for i in range(len(event_frames)) :
        if (event_contexts[i] == 'Left') :
            if (event_labels[i] == 'Foot_Strike_GS') :
                left_vector[left : event_frames[i]] = 0
            else :
                left_vector[left : event_frames[i]] = 1
        else :
            if (event_labels[i] == 'Foot_Strike_GS') :
                right_vector[right : event_frames[i]] = 0
            else :
                right_vector[right : event_frames[i]] = 1

    ## TODO: handle for the last indices from the last index to the end

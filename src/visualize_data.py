from read_data import *
import numpy as np
import itertools

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def show_graphic(label, files):
    # Displays (Y,Z) coordinates of each point of a file for one label in a graphic
    # for visual interpretation
    color_list = ['or', 'ob', 'og']

    legend_elements = [Line2D([0], [0], marker='o', color='r', label='Only right foot down'),
                       Line2D([0], [0], marker='o', color='b', label='Only left foot down'),
                       Line2D([0], [0], marker='o', color='g', label='Both feet down')]

    plt.figure()
    for file in files:
        x_disp, y_disp = get_2D_disp_data(label, file)
        y_disp = y_disp.astype(int) - 1

        color = np.take(color_list, y_disp)

        for i in range(len(y_disp)):
            plt.plot(x_disp[i, 1], x_disp[i, 0], color[i])

    plt.legend(handles=legend_elements)
    plt.title(label)
    plt.ylabel('Y coordinate (Vertical height)')
    plt.xlabel('Z coordinate (Complanar to T-pose human)')
    plt.show(block=False)



def show_labeled_x_speed(marker, file):
    color_list = ['or', 'ob', 'og']

    legend_elements = [Line2D([0], [0], marker='o', color='r', label='Only right foot down'),
                       Line2D([0], [0], marker='o', color='b', label='Only left foot down'),
                       Line2D([0], [0], marker='o', color='g', label='Both feet down')]


    # marker = 'LTOE'
    acq = get_acquisition_from_data(file)
    # X = acq.GetPoint(marker).GetValues()[:, 0]
    X, y = get_curve_data(marker, file)
    print(y)
    y = y.astype(int) #- 1
    print(y)
    print('\n\n')

    color = np.take(color_list, y)

    vitesse = []
    prev = X[0]
    for x in X[1:]:
        vitesse.append(round(x - prev, 3))
        prev = x

    plt.figure()
    for i in range(len(vitesse)):
        plt.plot(i, vitesse[i], color[i+1])

    plt.legend(handles=legend_elements)
    plt.ylabel("Marker's speed")
    plt.xlabel('Time')
    plt.title(marker + "-" + file)


def show_x_speed(marker, file):
    # marker = 'LTOE'
    acq = get_acquisition_from_data(file)
    X = acq.GetPoint(marker).GetValues()[:, 0]

    vitesse = []
    prev = X[0]
    for x in X[1:]:
        vitesse.append(round(x - prev, 3))
        prev = x

    plt.figure()
    # for i in range(len(vitesse)):
    #     plt.plot(i, vitesse[i], color[i+1])

    plt.plot(vitesse)
    plt.ylabel("Marker's speed")
    plt.xlabel('Time')
    plt.title(marker)


def get_curve_data(label, filename) :
    """Gets the magic numbers for our last method"""
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

    x = acq.GetPoint(label).GetValues()[start_frame:end_frame+1, 0]

    return x, vector

def get_speed_and_labels(files):
    # res = np.empty((0, 2), float)
    # res = []


    X1, y = get_curve_data('LTOE', files[0])
    X2, _ = get_curve_data('RTOE', files[0])

    v1 = []
    v2 = []
    prev1 = X1[0]
    prev2 = X2[0]
    for i in range(1, len(X1)):
        v1.append(round(X1[i] - prev1, 3))
        prev1 = X1[i]

        v2.append(round(X2[i] - prev2, 3))
        prev2 = X2[i]

    # print(np.array(v1).shape)
    # print(np.array(v2).shape)
    # print(np.array(y).shape)
    v = np.column_stack((v1, v2, y[1:])).T

    res = v[:-1]
    res_y = v[-1]

    # print(res.shape)
    # print(res_y.shape)

    for file in files[1:]:
        X1, y = get_curve_data('LTOE', file)
        X2, _ = get_curve_data('RTOE', file)

        v1 = []
        v2 = []
        prev1 = X1[0]
        prev2 = X2[0]
        for i in range(1, len(X1)):
            v1.append(round(X1[i] - prev1, 3))
            prev1 = X1[i]

            v2.append(round(X2[i] - prev2, 3))
            prev2 = X2[i]

        v = np.column_stack((v1, v2, y[1:])).T

        # print('adding')
        # print(v[:-1].shape)
        # print(v[-1].shape)
        res = np.column_stack((res, v[:-1]))

        res_y = np.append(res_y, v[-1])
        # print(X1[0:5])
        # print(X2[0:5])

    # print(res.shape)
    # print(res_y.shape)

    return res, res_y.T




    #     iters = [iter(v1), iter(v2)]
    #     X = np.array(list(it.__next__() for it in itertools.cycle(iters))) # Alternates coord from X1 then from X2 then back to X1 etc...
    #     print(X.shape)
    #
    #     res.extend(X) #= res + X#np.column_stack((res, X))
    # print(np.array(res).shape)


def get_2D_disp_data(label, filename) :
    """Used to get (Y,Z) data and associated class for a single file for data
    visualisation for kmeans"""
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

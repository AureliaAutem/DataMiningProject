from read_data import *

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

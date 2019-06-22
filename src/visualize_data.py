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

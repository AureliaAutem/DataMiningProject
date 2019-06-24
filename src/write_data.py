#Librairies
from btk import btk             #To read the data in .c3d files
import numpy as np
from read_data import *

def sub_x_coord(labels, file, to_sub):
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


def write_acq_to_file(acq, file):
     writer = btk.btkAcquisitionFileWriter()
     writer.SetInput(acq)
     writer.SetFilename(file)
     writer.Update()



def get_events_from_dense_model(model, X):
    """Predicts the labels for X using the trained model and extracts the events
    from it.
    Inputs :    - model : a trained model of the SimpleNeuralNetwork class. It
                    has to be trained with 'dense' data representation.
                - X : a list of input frames to classify. X has to be timely
                    ordered else it doesn't makes sens to use this function.
    Outputs :   - events : array of dictionnary containing "label", "context",
                    "frame" and "icon_id" key where each element of the array
                    represents a single event.
    """

    # Retrives the prediction of the model for input X
    y_pred = np.array(model.predict(X).tolist())
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)

    events = []
    saved_state = y_pred[0]
    for i in range(1, len(y_pred)):
        frame = y_pred[i]

        # Whenever we encounter a different state, we encountered an event
        if(saved_state != frame):
            if((saved_state == 2 and frame == 1) or (saved_state == 1 and frame == 2)):
                print('WARNING Illegal transition from state', saved_state, "to", frame)

            # Logic of the event extraction
            diff = saved_state - frame
            context = "Left" if (abs(diff) > 1) else "Right"
            isFootStrike = (diff < 0)
            label = "Foot_Strike_GS" if isFootStrike else "Foot_Off_GS"
            icon_id = 1 if isFootStrike else 2

            # Append the new event to the array
            events.append( {"frame": i, "context": context, "label": label, "icon_id": icon_id} )

            # Update previous state
            saved_state = frame

    print(events)
    print("total length:", len(events))
    return events

def get_events_from_sparse_model(model, X):
    """Predicts the labels for X using the trained model and extracts the events
    from it.
    Inputs :    - model : a trained model of the SimpleNeuralNetwork class. It
                    has to be trained with 'sparse' data representation.
                - X : a list of input frames to classify. X has to be timely
                    ordered else it doesn't makes sens to use this function.
    Outputs :   - events : array of dictionnary containing "label", "context",
                    "frame" and "icon_id" key where each element of the array
                    represents a single event.
    """

    # Retrives the prediction of the model for input X
    y_pred = np.array(model.predict(X).tolist())
    y_pred = np.argmax(y_pred, axis=1)
    print(y_pred)

    events = []
    for i in range(len(y_pred)):
        frame = y_pred[i]

        if(frame != 0):
            context = "Left" if (frame < 2) else "Right"
            isFootStrike = ((frame-1) % 2 == 0)
            label = "Foot_Strike_GS" if isFootStrike else "Foot_Off_GS"
            icon_id = 1 if isFootStrike else 2

            # Append the new event to the array
            events.append( {"frame": i, "context": context, "label": label, "icon_id": icon_id} )


    print(events)
    print("total length:", len(events))
    return events

def write_event_to_acq(events, acq):
    """Writes events to a given acq. Usualy to save the file later on.
    Inputs :    - events : array of dictionnary containing "label", "context"
                    and "frame" key when each element of the array represents
                    a single event.
                - acq : the aquisition file writer to write to
    Outputs :   - acq : the updated acq given as entry argument
    """

    # Delete existing events
    n = acq.GetEventNumber()
    # print("Writing events to acq")
    # print("Previously had", n, "events")
    for i in reversed(range(n)):
        acq.RemoveEvent(i)

    # Write new events
    for event in events:
        newEvent=btk.btkEvent() # build an empty event object
        newEvent.SetLabel(event["label"]) # set the label
        newEvent.SetContext(event["context"])
        # newEvent.SetFrame(event["frame"])
        newEvent.SetTime(event["frame"]/100)
        acq.AppendEvent(newEvent) # append the new event to the aquisition object

    # print("Now has", acq.GetEventNumber(), "events")
    return acq


def classify_video(model, labels, in_file, out_file, method):
    X_pred = get_prediction_X(labels, in_file)
    if(method == "dense"):
        events = get_events_from_dense_model(model, X_pred) # Only for dense representation of data
    elif (method == "sparse"):
        events = get_events_from_sparse_model(model, X_pred) # Only for sparse representation of data
    else:
        print("method '" + method + "' doesn't exist")
    acq_pred = get_acquisition_from_data(in_file)
    acq_pred = write_event_to_acq(events, acq_pred)
    write_acq_to_file(acq_pred, out_file)

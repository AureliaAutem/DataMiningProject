#Librairies
# from btk import btk             #To read the data in .c3d files
from read_data import *

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

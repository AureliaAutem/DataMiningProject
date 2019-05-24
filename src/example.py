from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np

from btk import btk
# http://biomechanical-toolkit.github.io/docs/Wrapping/Python/_getting_started.html
# http://python-future.org/compatible_idioms.html

# initialise
reader = btk.btkAcquisitionFileReader() 
reader.SetFilename("CP_GMFCS1_01916_20130128_18.c3d") 
reader.Update()
acq = reader.GetOutput()

# get some parameters
freq = acq.GetPointFrequency() # give the point frequency
print('freq : ', freq)
n_frames = acq.GetPointFrameNumber() # give the number of frames
print('n_frames : ', n_frames)
first_frame = acq.GetFirstFrame()
print('first_frame ', first_frame)

# metadata
metadata = acq.GetMetaData()

# events
n_events = acq.GetEventNumber()
print('n_event : ', n_events)
event = acq.GetEvent(0) # extract the first event of the aquisition
label = event.GetLabel() # return a string representing the Label
print('label : ', label)
context = event.GetContext() # return a string representing the Context
print('context : ', context)
event_frame = event.GetFrame() # return the frame as an integer
print('event_frame : ', event_frame)

# get points
point_labels = metadata.FindChild("POINT").value().FindChild("LABELS").value().GetInfo().ToString()
print('labels :', point_labels)
points = acq.GetPoints().GetItemNumber()
print('points : ', points)

# exemple on how to construct array with markers from one frame
Frame = 0
data_FrameChosen = np.array([acq.GetPoint('LASI').GetValues()[Frame,:],
                            acq.GetPoint('RASI').GetValues()[Frame,:], 
                            acq.GetPoint('LPSI').GetValues()[Frame,:], 
                            acq.GetPoint('RPSI').GetValues()[Frame,:]])

print('shape : ', data_FrameChosen.shape)
print('array : ', data_FrameChosen)

# generalasize to get an array of points for a set of frames
# get markers
markers = list()
start = False
for label in point_labels:
    label = label.replace(' ', '')
    if label == 'C7':
        start = True
    if label == 'CentreOfMass':
        break
    if start:
        markers.append(label)

print(len(markers))
print(markers)

# get events
n_events = acq.GetEventNumber()
event_frames = [acq.GetEvent(event).GetFrame() for event in range(n_events)]
event_frames.sort()
print(event_frames)
start_frame = event_frames[0]-first_frame
end_frame = event_frames[-1]-first_frame
print(start_frame, end_frame)

# get data for each marker
data = [[acq.GetPoint(marker).GetValues()[frame,:] for marker in markers] for frame in range(start_frame, end_frame+1)]
data = np.array(data)
print(data.shape)
print(np.count_nonzero(np.isnan(data)))

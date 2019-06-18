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
print('##### Information about the file #####')
freq = acq.GetPointFrequency() # give the point frequency
print('Frequency : ', freq)
n_frames = acq.GetPointFrameNumber() # give the number of frames
print('Number of frames : ', n_frames)
first_frame = acq.GetFirstFrame()
print('First frame ', first_frame)

# metadata
metadata = acq.GetMetaData()

# events
print('\n\n##### Information about one event #####')

n_events = acq.GetEventNumber()
print('Number of events : ', n_events)
event = acq.GetEvent(0) # extract the first event of the aquisition
label = event.GetLabel() # return a string representing the Label
print('First event label : ', label)
context = event.GetContext() # return a string representing the Context
print('Fisrt event context : ', context)
event_frame = event.GetFrame() # return the frame as an integer
print('First event frame : ', event_frame)

# get points
print('\n\n##### Information about the labels #####')
point_labels = metadata.FindChild("POINT").value().FindChild("LABELS").value().GetInfo().ToString()
print('All labels :', point_labels)
points = acq.GetPoints().GetItemNumber()
print('Number of points : ', points)

# exemple on how to construct array with markers from one frame
Frame = 0
data_FrameChosen = np.array([acq.GetPoint('LASI').GetValues()[Frame,:],
                            acq.GetPoint('RASI').GetValues()[Frame,:],
                            acq.GetPoint('LPSI').GetValues()[Frame,:],
                            acq.GetPoint('RPSI').GetValues()[Frame,:]])

print('\nWe extract the first frame and its points : LASI, RASI, LPSI, RPSI')
print('Shape : ', data_FrameChosen.shape)
print('Array : ', data_FrameChosen)

# # generalasize to get an array of points for a set of frames
# # get markers
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
#
# print(len(markers))
# print(markers)

# get events
print('\n\n##### Information about events #####')

n_events = acq.GetEventNumber()
event_frames = [acq.GetEvent(event).GetFrame() for event in range(n_events)]
event_frames.sort()
print("Frame of the 4 events : ", event_frames)
start_frame = event_frames[0]-first_frame
end_frame = event_frames[-1]-first_frame
print("Interesting frames : ", start_frame, end_frame)

# get data for each marker
print('\n\n##### Information about data for each marker #####')
data = [[acq.GetPoint(marker).GetValues()[frame,:] for marker in markers] for frame in range(start_frame, end_frame+1)]
data = np.array(data)
print("Shape of the array : (nb_frames, nb_markers, dim)", data.shape)
#print(np.count_nonzero(np.isnan(data)))

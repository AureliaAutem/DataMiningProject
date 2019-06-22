#Librairies
from btk import btk             #To read the data in .c3d files
import numpy as np
from read_data import get_acquisition_from_data

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

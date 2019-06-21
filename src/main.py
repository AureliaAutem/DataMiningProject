
#Librairies
import numpy as np              #To deal with arrays in python
import glob                     #To read filenames in folder

#Our own functions
from read_data import *
from manipulation import *
from write_data import *

#We read all the data filenames
root = "data/"
labels = ["RTOE", "RANK", "RHEE", "T10", "LTOE", "LANK", "LHEE"]

#We separate the data
# (train, test) = split_train_test_files(root)
#
# X_train = get_data_by_labels(labels, train, method="dense")
# X_test = get_data_by_labels(labels, test, method="dense")
#
# print(X_train.shape)
# print(X_test.shape)

reader = btk.btkAcquisitionFileReader()
reader.SetFilename("CP_GMFCS1_01916_20130128_18.c3d")
reader.Update()
acq = reader.GetOutput()
metadata = acq.GetMetaData()
point_labels = metadata.FindChild("POINT").value().FindChild("LABELS").value().GetInfo().ToString()


acq = sub_x_coord(point_labels, 'CP_GMFCS1_01916_20130128_18.c3d', 'T10')
write_acq(acq)

#reading_data(filenames[0])
# for i in range (len(filenames)) :
#     print_infos(extract_infos(filenames[i], foldername))

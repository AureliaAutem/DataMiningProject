
#Librairies
import numpy as np              #To deal with arrays in python
import glob                     #To read filenames in folder

#Our own functions
from read_data import *
from manipulation import *

#We read all the data filenames
root = "data/"
labels = ["LASI", "RASI", "LPSI", "RPSI"]

#We separate the data
(train, test) = split_train_test_files(root)

X_train = get_data_by_labels(labels, train, method="dense")
X_test = get_data_by_labels(labels, test, method="dense")

print(X_train.shape)
print(X_test.shape)


#reading_data(filenames[0])
# for i in range (len(filenames)) :
#     print_infos(extract_infos(filenames[i], foldername))

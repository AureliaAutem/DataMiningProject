
#Librairies
import numpy as np              #To deal with arrays in python
import glob                     #To read filenames in folder

#Our own functions
from read_data import *
from manipulation import *

#We read all the data filenames
foldername = "data/"
filenames = glob.glob(foldername+"*.c3d")
print(filenames)
print(len(filenames))

#reading_data(filenames[0])
for i in range (len(filenames)) :
    print_infos(extract_infos(filenames[i], foldername))

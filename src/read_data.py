#Librairies
from btk import btk             #To read the data in .c3d files

def get_acquisition_from_data(file) :

    # We create a new acquisition object
    reader = btk.btkAcquisitionFileReader() # build a btk reader object
    reader.SetFilename(file) # set a filename to the reader
    reader.Update()
    acq = reader.GetOutput() # acq is the btk aquisition object
    return acq

def print_infos(file) :
    #We extract the data
    acq = get_acquisition_from_data(file)

    

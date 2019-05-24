def extract_infos(filename, foldername) :
    infos = {}
    filename = filename[len(foldername):]

    #Finding the pathology    
    (string1, filename) = get_string_to_next_underscore(filename)
    (string2, filename) = get_string_to_next_underscore(filename)

    if (isID(string2)) :
        infos["pathology"] = string1
        infos["ID_patient"] = string2
    else :
        infos["pathology"] = string1 + "_" + string2
        (string, filename) = get_string_to_next_underscore(filename)
        infos["ID_patient"] = string

    (string, filename) = get_string_to_next_underscore(filename)
    infos["date"] = string
    infos["ID_trial"] = filename[:2]
    
    return infos

def print_infos(infos) :
    print("Pathology name :\t" + infos["pathology"])
    print("ID patient :\t\t" + infos["ID_patient"])
    print("Date of the trial:\t" + infos["date"][-2:] + "." + infos["date"][4:6] + "." + infos["date"][:4])
    print("ID of the trial :\t" + infos["ID_trial"])


def get_string_to_next_underscore(string) :
    for i in range (len(string)) :
        if (string[i] == "_") : break

    return (string[:i], string[i+1:])

def isID(string) :
    for elt in string :
        try :
            int(elt)
        except :
            return False
    return True

import os


def renameFiles():
    ## Add IN- / OUT- prefix to .txt file.
    path = 'FCC-21X21/disNodes2/'
    for file in os.scandir(path):
        if file.name.endswith('.txt') and 'IN-' not in file.name and 'OUT-' not in file.name:
            #print(path + file.name)
            os.rename(path+file.name, f'{path}OUT-{file.name}')
        else:
            pass


    ## Change .txt file to .csv file.
    path = 'FCC-21X21/disNodes2/'
    for file in os.scandir(path):
        if file.name.endswith('.txt') and 'IN-' not in file.name and 'OUT-' not in file.name:
            #print(path + file.name)
            os.rename(path+file.name, f'{path}OUT-{file.name}')
        else:
            pass


    ## Change numbering.
    path = 'Ti/tri-old/'
    for file in os.scandir(path):
        if file.name.endswith('.csv') and 'IN-' not in file.name and 'OUT-' not in file.name:
            #print(path + file.name)
            os.rename(path+file.name, f'{path}OUT-{file.name}')
        else:
            passs
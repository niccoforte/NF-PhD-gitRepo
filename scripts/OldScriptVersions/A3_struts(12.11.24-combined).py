import os
import numpy as np

os.chdir("C:\\Users\\exy053\\Documents\\validation\\10\\FCC")

def export_struts(inpFile, expFile):
    with open(inpFile, 'r') as f:
        lines = f.readlines()

    strut_lines = [lines[lines.index(line)+1].split(' ') for line in lines if "*Beam Section, elset=" in line]
    thicks = [float(line[-1].strip('\n')) for line in strut_lines]
    thicks_check = list(set(thicks))
    thicks_check.sort(reverse = True)
    if len(thicks_check) == 2:
        if round(thicks_check[0],3) == round(2*thicks_check[1],3):
            thicks = [t for t in thicks if t != thicks_check[0]]
    elif len(thicks_check) > 2:
        if round(thicks_check[0],1) == 2*round(np.mean(thicks_check[1:]),1):
            thicks = [t for t in thicks if t != thicks_check[0]]
        
    with open(expFile, 'w') as f:
        for thick in thicks:
            f.write(str(thick) + '\n')


if not os.path.exists("transfer"):
    os.makedirs("transfer")

for file in os.scandir():
    if 'per' in file.name or 'disStruts' in file.name:
        if file.name.endswith('.inp'):
            expFile = "transfer/IN-s" + file.name[:-4].replace('_','-') + ".csv"
            export_struts(file.name, expFile)

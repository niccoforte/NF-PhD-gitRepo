from abaqus import *
from abaqusConstants import *
from caeModules import *

import os
from os.path import join, getsize

parentDirectory = r'C:\\Users\\exy053\\Documents'

testMode = False
deleteOldOdbs = True

suffix = str(minorVersion)
for curDirectory, folders, files in os.walk(parentDirectory):
    print('\n*** Reading directory: {}'.format(curDirectory))
    odbCount = 0
    odbs = [f for f in files if f.endswith('.odb')]
    for odb in odbs:
        odbPath = join(curDirectory, odb)
        try:
            if session.isUpgradeRequiredForOdb(odbPath):
                newOdbPath = odbPath.replace('.odb', '_' + suffix + '.odb')
                if not testMode:
                    session.upgradeOdb(existingOdbPath=odbPath,
                                       upgradedOdbPath=newOdbPath)
                    if deleteOldOdbs:
                        os.remove(odbPath)
                        os.rename(newOdbPath, odbPath)
                odbSizeMB = float(getsize(odbPath))/(1024. * 1024.)
                print('Upgraded: {0:s} ({1:.2f} MB)'.format(odbPath, odbSizeMB))
                
                odbCount += 1
                
        except Exception as e:
            print('Could not be read/upgraded: {0:s}'.format(odbPath))
    print('+ Upgraded {0:d} odbs'.format(odbCount))
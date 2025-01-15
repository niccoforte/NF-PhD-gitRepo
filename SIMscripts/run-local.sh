#!/bin/bash

LAT=tri
DIS=per
nnx=6
unitCellSize=10
mode=fracture
material=ti
rD=0.4
initial=1
nJobs=1
CPUs=12
Fout=100
Hout=200
dir=0								  # val, size, sic, rd, modelchanges, input
strainApp=a

cd C:/Users/exy053/Docoments

abaqus cae noGUI=A1_FractureToughness-Ductility.py -- $LAT $DIS $nnx $unitCellSize $mode $material $rD $initial $nJobs $CPUs $Fout $Hout $dir
abaqus cae noGUI=A2_Ductile-postProcess.py -- $LAT $DIS $nnx $unitCellSize $mode $material $rD $initial $nJobs $CPUs $Fout $Hout $dir
abaqus cae noGUI=A2_Fracture-postProcess.py -- $LAT $DIS $nnx $unitCellSize $mode $material $rD $initial $nJobs $CPUs $Fout $Hout $dir

python A3_nodes.py
python A3_struts.py

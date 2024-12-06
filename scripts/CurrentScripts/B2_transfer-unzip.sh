#!/bin/bash

LAT="tri"
nnx=30
DIS="disNodes"
num="4"
JOB="3326037"

cd C:/Users/exy053
cd 'OneDrive - Queen Mary, University of London'
cd Documents/Research/Paper1-LatticeFractureToughness/Lattice-Fracture-Toughness/PINN/data/$LAT-$nnx/$DIS$num

/bin/echo Running on host: `hostname`.
/bin/echo Starting on: `date`, in `pwd`.

scp exy053@andrena.hpc.qmul.ac.uk:/data/home/exy053/Paper1-LatticeFractureToughness/sims/$LAT/C1_transfer-$LAT-$DIS-$JOB.tgz "`pwd`"
tar -xvzf C1_transfer-$LAT-$DIS-$JOB.tgz

cp data/scratch/exy053/$LAT/$DIS-$JOB/transfer/* "`pwd`"

rm -rf data
rm -rf C1_transfer-$LAT-$DIS-$JOB.tgz

#scp exy053@andrena.hpc.qmul.ac.uk:/data/SEMS-TaoLab/Niccolo-Forte/Al/C2_$LAT-$DIS-$JOB.tgz "`pwd`"
#tar -xvzf C2_$LAT-$DIS-$JOB.tgz

#read -p "Press any key to continue" x

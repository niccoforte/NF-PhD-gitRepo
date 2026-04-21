#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8
#$ -l h_rt=240:0:0
#$ -l h_vmem=5G
# -l gpu=2
# -l cluster=andrena
# -l highmem
#$ -N JobNameOG
#$ -l abaqus=12
# -l avx512

set -e
USER=exy053

# ^^^ RENAME ^^^
LAT=lat
nnx=10
unitCellSize=10
mode=both
material=ti
rD=0.2
DIS=per
fac=0.0
distribution=lhs_uniform
target=all
initial=1
nJobs=1
CPUs=$NSLOTS
Fout=20
Hout=200
pDir=None

PATH_EXTRA=

zip=false
delete_scratch=true


# Load required modules
module load abaqus/2024
module load intel

/bin/echo Running on host: `hostname`.
/bin/echo Starting on: `date`, in `pwd`.


# make dir
mkdir /data/scratch/$USER/$JOB_ID
mkdir /data/scratch/$USER/$JOB_ID/transfer/
mkdir /data/scratch/$USER/$JOB_ID/zip/
mkdir /data/scratch/$USER/$JOB_ID/zip/transfer/


# copy command
rsync -av /data/home/exy053/p1/p1git-Lattices/SIMscripts/A-HPC-* /data/scratch/$USER/$JOB_ID
rsync -av $SGE_O_WORKDIR/B* /data/scratch/$USER/$JOB_ID
cd /data/scratch/$USER/$JOB_ID

/bin/echo Working in directory: `pwd`.


# Replace the following line with abaqus command
#abaqus job=callmat cpus=${NSLOTS} user=PhaseField_5m-std.o mp_mode=THREADS scratch=/data/scratch/$USER/
abaqus cae noGUI=A-HPC-1_FractureToughness-Ductility.py -- $LAT $nnx $unitCellSize $mode $material $rD $DIS $fac $distribution $target $initial $nJobs $CPUs $Fout $Hout $pDir

/bin/echo Simulation completed at: `date`.
/bin/echo Processing outputs...

abaqus cae noGUI=A-HPC-2_OUTpostProcess.py -- $LAT $nnx $unitCellSize $mode $material $rD $DIS $fac $distribution $target $initial $nJobs $CPUs $Fout $Hout $pDir
abaqus cae noGUI=A-HPC-2_INpostProcess.py -- $LAT $nnx $unitCellSize $mode $material $rD $DIS $fac $distribution $target $initial $nJobs $CPUs $Fout $Hout $pDir

/bin/echo Inputs and outputs collected.


# file transfer
rsync -av /data/scratch/$USER/$JOB_ID/A* /data/scratch/$USER/$JOB_ID/zip/
rsync -av /data/scratch/$USER/$JOB_ID/B* /data/scratch/$USER/$JOB_ID/zip/
rsync -av /data/scratch/$USER/$JOB_ID/abaqus* /data/scratch/$USER/$JOB_ID/zip/
rsync -av /data/scratch/$USER/$JOB_ID/*.odb /data/scratch/$USER/$JOB_ID/zip/
rsync -av /data/scratch/$USER/$JOB_ID/*.inp /data/scratch/$USER/$JOB_ID/zip/
rsync -av /data/scratch/$USER/$JOB_ID/transfer/* /data/scratch/$USER/$JOB_ID/zip/transfer/

/bin/echo Simulation files in /data/scratch/$USER/$JOB_ID/zip.


# clean up and compression
rsync -av $SGE_O_WORKDIR/$JOB_NAME.o$JOB_ID /data/scratch/$USER/$JOB_ID/
rsync -av $SGE_O_WORKDIR/$JOB_NAME.o$JOB_ID /data/scratch/$USER/$JOB_ID/zip/

mkdir /data/SEMS-TaoLab/Niccolo-Forte/Ti/data/$DIS/$PATH_EXTRA/$fac/$LAT/$JOB_ID/
mkdir /data/SEMS-TaoLab/Niccolo-Forte/Ti/data/$DIS/$PATH_EXTRA/$fac/$LAT/$JOB_ID/zip/
rsync -av /data/scratch/$USER/$JOB_ID/zip/* /data/SEMS-TaoLab/Niccolo-Forte/Ti/data/$DIS/$PATH_EXTRA/$fac/$LAT/$JOB_ID/zip/

if [ "$zip" = true ] ; then
    tar -czf C1_transfer-$LAT-$DIS-$JOB_ID.tgz /data/scratch/$USER/$JOB_ID/transfer/
    tar -czf C2_zip-$LAT-$DIS-$JOB_ID.tgz /data/scratch/$USER/$JOB_ID/zip/
    rsync -av /data/scratch/$USER/$JOB_ID/C1_transfer-$LAT-$DIS-$JOB_ID.tgz $SGE_O_WORKDIR
    rsync -av /data/scratch/$USER/$JOB_ID/C2_zip-$LAT-$DIS-$JOB_ID.tgz /data/SEMS-TaoLab/Niccolo-Forte/Ti/data/$DIS/$PATH_EXTRA/$fac/$LAT/
fi

if [ "$delete_scratch" = true ] ; then
    rm -rf /data/scratch/$USER/$JOB_ID
fi

/bin/echo Job completed at: `date`.
/bin/echo Finished. Data saved in $SGE_O_WORKDIR and /data/SEMS-TaoLab/Niccolo-Forte/

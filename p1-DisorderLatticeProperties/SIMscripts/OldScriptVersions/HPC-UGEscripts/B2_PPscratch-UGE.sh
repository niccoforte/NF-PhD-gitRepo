#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 4
#$ -l h_rt=1:0:0
#$ -l h_vmem=2G
# -l gpu=2
# -l cluster=andrena
# -l highmem
#$ -N LATppScatch
# -l abaqus=12
# -l avx512

set -e

# ^^^ RENAME ^^^
LAT=kagome
DIS=disNodes
fac=0.2
nnx=20
unitCellSize=10
rD=0.2
initial=101
nJobs=200
CPUs=$NSLOTS

zip=false
delete_scratch=true

ppJOB=5774827
ppJOBparent=/data/home/exy053/p1/Ti/$DIS/$fac/$LAT


# Load required modules
module load abaqus/2024
module load intel

/bin/echo Running on host: `hostname`.
/bin/echo Starting on: `date`, in `pwd`.

/bin/echo Working in directory: `pwd`.

# Replace the following line with abaqus command
abaqus cae noGUI=A-HPC-2_OUTpostProcess.py -- $LAT $DIS $fac $nnx $unitCellSize $rD $initial $nJobs $CPUs
abaqus python A-HPC-2_INpostProcess.py -- $LAT $DIS $fac $nnx $unitCellSize $rD $initial $nJobs $CPUs

/bin/echo Inputs and outputs collected.


# file transfer
rsync -av /data/scratch/exy053/$ppJOB/A* /data/scratch/exy053/$ppJOB/zip/
rsync -av /data/scratch/exy053/$ppJOB/B* /data/scratch/exy053/$ppJOB/zip/
rsync -av /data/scratch/exy053/$ppJOB/abaqus* /data/scratch/exy053/$ppJOB/zip/
rsync -av /data/scratch/exy053/$ppJOB/*.odb /data/scratch/exy053/$ppJOB/zip/
rsync -av /data/scratch/exy053/$ppJOB/*.inp /data/scratch/exy053/$ppJOB/zip/
rsync -av /data/scratch/exy053/$ppJOB/transfer/* /data/scratch/exy053/$ppJOB/zip/transfer/

/bin/echo Simulation files in /data/scratch/exy053/$ppJOB/zip.


# clean up and compression
rsync -av $ppJOBparent/*.o$ppJOB /data/scratch/exy053/$ppJOB/zip
rsync -av $SGE_O_WORKDIR/$JOB_NAME.o$JOB_ID /data/scratch/exy053/$ppJOB/zip/

mkdir /data/SEMS-TaoLab/Niccolo-Forte/Ti/data/$DIS/$fac/$LAT/$ppJOB/
mkdir /data/SEMS-TaoLab/Niccolo-Forte/Ti/data/$DIS/$fac/$LAT/$ppJOB/zip/
rsync -av /data/scratch/exy053/$ppJOB/zip/* /data/SEMS-TaoLab/Niccolo-Forte/Ti/data/$DIS/$fac/$LAT/$ppJOB/zip/

if [ "$zip" = true ] ; then
    tar -czf C1_transfer-$LAT-$DIS-$ppJOB.tgz /data/scratch/exy053/$ppJOB/transfer/
    tar -czf C2_zip-$LAT-$DIS-$ppJOB.tgz /data/scratch/exy053/$ppJOB/zip/
    rsync -av /data/scratch/exy053/$ppJOB/C1_transfer-$LAT-$DIS-$ppJOB.tgz $SGE_O_WORKDIR
    rsync -av /data/scratch/exy053/$ppJOB/C2_zip-$LAT-$DIS-$ppJOB.tgz /data/SEMS-TaoLab/Niccolo-Forte/Ti/data/$DIS/$fac/$LAT/
fi

if [ "$delete_scratch" = true ] ; then
    rm -rf /data/scratch/exy053/$ppJOB
fi

/bin/echo Job completed at: `date`.
/bin/echo Finished. Data saved in $SGE_O_WORKDIR and /data/SEMS-TaoLab/Niccolo-Forte/

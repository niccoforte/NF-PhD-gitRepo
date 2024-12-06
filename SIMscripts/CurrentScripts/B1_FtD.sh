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

# ^^^ RENAME ^^^
LAT=lat
DIS=per
nnx=10
unitCellSize=10
rD=0.2
initial=1
nJobs=100
CPUs=$NSLOTS


# Load required modules
module load abaqus/2020
module load intel

/bin/echo Running on host: `hostname`.
/bin/echo Starting on: `date`, in `pwd`.


# make dir
mkdir /data/scratch/exy053/$JOB_ID
mkdir /data/scratch/exy053/$JOB_ID/transfer/
mkdir /data/scratch/exy053/$JOB_ID/zip/
mkdir /data/scratch/exy053/$JOB_ID/zip/transfer/


# copy command
rsync -av $SGE_O_WORKDIR/A* /data/scratch/exy053/$JOB_ID
rsync -av $SGE_O_WORKDIR/B* /data/scratch/exy053/$JOB_ID
cd /data/scratch/exy053/$JOB_ID

/bin/echo Working in directory: `pwd`.


# Replace the following line with abaqus command
#abaqus job=callmat cpus=${NSLOTS} user=PhaseField_5m-std.o mp_mode=THREADS scratch=/data/scratch/exy053/
#abaqus job=TPB_honey_Nx96.inp cpus=${NSLOTS} user=PhaseField_call_mat_AT1_D0.for mp_mode=THREADS scratch=/data/scratch/exy053/
#abaqus job=Cantilever.inp cpus=${NSLOTS} user=vumat.for mp_mode=THREADS scratch=/data/scratch/exy053/ memory="90 %" interactive
#abaqus job=Job-b cpus=16 user=USDFLDTest3.f mp_mode=THREADS scratch=/data/scratch/exy053/ memory="90 %" interactive

abaqus cae noGUI=A1_FractureToughness-Ductility.py -- $LAT $DIS $nnx $unitCellSize $rD $initial $nJobs $CPUs

/bin/echo Simulation completed at: `date`.
/bin/echo Processing outputs...

abaqus cae noGUI=A2_OUTpostProcess.py -- $LAT $DIS $nnx $unitCellSize $rD $initial $nJobs $CPUs

module load python
python A2_INpostProcess.py -- $LAT $DIS $nnx $unitCellSize $rD $initial $nJobs $CPUs

/bin/echo Inputs and outputs collected.


# file transfer
rsync -av /data/scratch/exy053/$JOB_ID/A* /data/scratch/exy053/$JOB_ID/zip/
rsync -av /data/scratch/exy053/$JOB_ID/B* /data/scratch/exy053/$JOB_ID/zip/
rsync -av /data/scratch/exy053/$JOB_ID/abaqus* /data/scratch/exy053/$JOB_ID/zip/
rsync -av /data/scratch/exy053/$JOB_ID/*.odb /data/scratch/exy053/$JOB_ID/zip/
rsync -av /data/scratch/exy053/$JOB_ID/*.inp /data/scratch/exy053/$JOB_ID/zip/
rsync -av /data/scratch/exy053/$JOB_ID/transfer/* /data/scratch/exy053/$JOB_ID/zip/transfer/

/bin/echo Simulation files in /data/scratch/exy053/$JOB_ID/zip.


# clean up and compression
rsync -av $SGE_O_WORKDIR/$JOB_NAME.o$JOB_ID /data/scratch/exy053/$JOB_ID/
rsync -av $SGE_O_WORKDIR/$JOB_NAME.o$JOB_ID /data/scratch/exy053/$JOB_ID/zip/

tar -czf C1_transfer-$LAT-$DIS-$JOB_ID.tgz /data/scratch/exy053/$JOB_ID/transfer/
tar -czf C2_zip-$LAT-$DIS-$JOB_ID.tgz /data/scratch/exy053/$JOB_ID/zip/

mkdir $SGE_O_WORKDIR/$JOB_ID/
mkdir $SGE_O_WORKDIR/$JOB_ID/zip/
rsync -av /data/scratch/exy053/$JOB_ID/zip/* $SGE_O_WORKDIR/$JOB_ID/zip/
rsync -av /data/scratch/exy053/$JOB_ID/C1_transfer-$LAT-$DIS-$JOB_ID.tgz $SGE_O_WORKDIR
rsync -av /data/scratch/exy053/$JOB_ID/C2_zip-$LAT-$DIS-$JOB_ID.tgz /data/SEMS-TaoLab/Niccolo-Forte/Ti/data/

rm -rf /data/scratch/exy053/$JOB_ID

/bin/echo Job completed at: `date`.
/bin/echo Final and compressed data saved in $SGE_O_WORKDIR and /data/SEMS-TaoLab/Niccolo-Forte/

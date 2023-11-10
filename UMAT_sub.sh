#!/bin/bash
#$ -pe smp 16
#$ -l h_vmem=2G
# -l highmem
#$ -l h_rt=24:0:0
#$ -cwd
#$ -j y
#$ -N abq-job
#$ -l abaqus=16

# Load the required modules
module load abaqus/2020
module load intel

/bin/echo Running on host: `hostname`.
/bin/echo Starting on: `date`.

# make dir
mkdir /data/scratch/exy053/$JOB_ID
mkdir $SGE_O_WORKDIR/$JOB_ID

# copy command
cp $SGE_O_WORKDIR/* /data/scratch/exy053/$JOB_ID
cd /data/scratch/exy053/$JOB_ID
/bin/echo In directory: `pwd`

# Replace the following line with abaqus command
#abaqus job=callmat cpus=${NSLOTS} user=PhaseField_5m-std.o mp_mode=THREADS scratch=/data/scratch/exy053/

#abaqus job=TPB_honey_Nx96.inp cpus=${NSLOTS} user=PhaseField_call_mat_AT1_D0.for mp_mode=THREADS scratch=/data/scratch/exy053/
#abaqus job=Cantilever.inp cpus=${NSLOTS} user=vumat.for mp_mode=THREADS scratch=/data/scratch/exy053/ memory="90 %" interactive


abaqus job=Job-b cpus=16 user=USDFLDTest3.f mp_mode=THREADS scratch=/data/scratch/exy053/ memory="90 %" interactive


#sleep 300000

echo Now it is: `date`

# copy back
mkdir $SGE_O_WORKDIR/$JOB_ID
cp /data/scratch/exy053/$JOB_ID/* $SGE_O_WORKDIR/$JOB_ID
#cd /data/scratch/exy053/
#cp /data/scratch/exy053/$JOB_ID/* /data/home/exy053/
# clean up
rm -rf /data/scratch/exy053/$JOB_ID


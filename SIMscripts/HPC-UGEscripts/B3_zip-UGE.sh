#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 8
#$ -l h_rt=1:0:0
#$ -l h_vmem=2G
# -l gpu=2
# -l cluster=andrena
# -l highmem
#$ -N reZIP
# -l avx512

set -e

# ^^^ RENAME ^^^
LAT=lat
DIS=per
JOBreZIP=1111111


# Load required modules
module load intel

/bin/echo Running on host: `hostname`.
/bin/echo Starting on: `date`, in `pwd`.


# make dir
mkdir /data/scratch/exy053/$JOBreZIP
mkdir /data/scratch/exy053/$JOBreZIP/zip/
mkdir /data/scratch/exy053/$JOBreZIP/zip/transfer/


# copy command
rsync -av $SGE_O_WORKDIR/zip/* /data/scratch/exy053/$JOBreZIP/zip/
rsync -av $SGE_O_WORKDIR/zip/transfer/* /data/scratch/exy053/$JOBreZIP/zip/transfer/
cd /data/scratch/exy053/$JOBreZIP

/bin/echo Working in directory: `pwd`.

tar -czf C1_transfer-$LAT-$DIS-$JOBreZIP.tgz /data/scratch/exy053/$JOBreZIP/zip/transfer/
tar -czf C2_zip-$LAT-$DIS-$JOBreZIP.tgz /data/scratch/exy053/$JOBreZIP/zip/

rsync -av C1_transfer-$LAT-$DIS-$JOBreZIP.tgz /data/home/exy053/Paper1-LatticeFractureToughness/Ti/$LAT/
rsync -av C2_zip-$LAT-$DIS-$JOBreZIP.tgz /data/SEMS-TaoLab/Niccolo-Forte/Ti/data/

rm -rf /data/scratch/exy053/$JOBreZIP

/bin/echo Job completed at: `date`.

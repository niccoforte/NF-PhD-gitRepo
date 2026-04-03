#!/bin/bash
#$ -cwd
#$ -j y
#$ -pe smp 4
#$ -l h_rt=240:0:0
#$ -l h_vmem=11G
# -l gpu=2
# -l cluster=andrena
# -l highmem
#$ -N move

set -e

# INPUT
LAT="tri-new"
DIS="20dN"
JOB_ID="4*"


/bin/echo Running on host: `hostname`.
/bin/echo Starting on: `date`, in `pwd`.


#rsync -av /data/home/exy053/p1/Ti/$DIS/$LAT/* /data/SEMS-TaoLab/Niccolo-Forte/Ti/data/$DIS/$LAT/
#rm -rf /data/home/exy053/p1/Ti/$DIS/$LAT/$JOB_ID
#rm -rf /data/home/exy053/p1/Ti/$DIS/$LAT/C1*


# ==================================


#cd /data/scratch/exy053/$LAT/$DIS-$JOB_ID

#tar -czf C1_transfer-$LAT-$DIS-$JOB_ID.tgz /data/scratch/exy053/$JOB_ID/transfer/
#tar -czf C2_zip-$LAT-$DIS-$JOB_ID.tgz /data/scratch/exy053/$JOB_ID/zip/

#rsync -av /data/scratch/exy053/$JOB_ID/C1_transfer-$LAT-$DIS-$JOB_ID.tgz /data/home/exy053/Paper1-LatticeFractureToughness/sims/$LAT/
#rsync -av /data/scratch/exy053/$JOB_ID/C2_zip-$LAT-$DIS-$JOB_ID.tgz /data/SEMS-TaoLab/Niccolo-Forte/Ti/

#rm -rf /data/scratch/exy053/$JOB_ID

#/bin/echo Job completed at: `date`.
#/bin/echo Compressed data saved in /data/home/exy053/Paper1-LatticeFractureToughness/sims/$LAT/ and /data/SEMS-TaoLab/Niccolo-Forte/


# ===================================


#cd /data/scratch/exy053/FCC/disNodes-3326034
#tar -czf C2_FCC-disNodes-3326034.tgz /data/scratch/exy053/FCC/disNodes-3326034/
#rsync -av /data/scratch/exy053/FT-StrutElemConv /data/SEMS-TaoLab/Niccolo-Forte/Ti
#!/bin/bash
# -j y
#SBATCH -n 8
#SBATCH -p compute
#SBATCH -t 240:0:0
#SBATCH --mem-per-cpu=5G
# -l gpu=2
# -l cluster=andrena
# -l highmem
#SBATCH -N JobNameOG
#SBATCH -l abaqus=12
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
mkdir /gpfs/scratch/$USER/$SLURM_JOB_ID
mkdir /gpfs/scratch/$USER/$SLURM_JOB_ID/transfer/
mkdir /gpfs/scratch/$USER/$SLURM_JOB_ID/zip/
mkdir /gpfs/scratch/$USER/$SLURM_JOB_ID/zip/transfer/


# copy command
rsync -av /data/home/exy053/p1/p1git-Lattices/SIMscripts/A-HPC-* /gpfs/scratch/$USER/$SLURM_JOB_ID
rsync -av $SGE_O_WORKDIR/B* /gpfs/scratch/$USER/$SLURM_JOB_ID
cd /gpfs/scratch/$USER/$SLURM_JOB_ID

/bin/echo Working in directory: `pwd`.


# Replace the following line with abaqus command
#abaqus job=callmat cpus=${NSLOTS} user=PhaseField_5m-std.o mp_mode=THREADS scratch=/gpfs/scratch/$USER/
abaqus cae noGUI=A-HPC-1_FractureToughness-Ductility.py -- $LAT $nnx $unitCellSize $mode $material $rD $DIS $fac $distribution $target $initial $nJobs $CPUs $Fout $Hout $pDir

/bin/echo Simulation completed at: `date`.
/bin/echo Processing outputs...

abaqus cae noGUI=A-HPC-2_OUTpostProcess.py -- $LAT $nnx $unitCellSize $mode $material $rD $DIS $fac $distribution $target $initial $nJobs $CPUs $Fout $Hout $pDir
abaqus cae noGUI=A-HPC-2_INpostProcess.py -- $LAT $nnx $unitCellSize $mode $material $rD $DIS $fac $distribution $target $initial $nJobs $CPUs $Fout $Hout $pDir

/bin/echo Inputs and outputs collected.


# file transfer
rsync -av /gpfs/scratch/$USER/$SLURM_JOB_ID/A* /gpfs/scratch/$USER/$SLURM_JOB_ID/zip/
rsync -av /gpfs/scratch/$USER/$SLURM_JOB_ID/B* /gpfs/scratch/$USER/$SLURM_JOB_ID/zip/
rsync -av /gpfs/scratch/$USER/$SLURM_JOB_ID/abaqus* /gpfs/scratch/$USER/$SLURM_JOB_ID/zip/
rsync -av /gpfs/scratch/$USER/$SLURM_JOB_ID/*.odb /gpfs/scratch/$USER/$SLURM_JOB_ID/zip/
rsync -av /gpfs/scratch/$USER/$SLURM_JOB_ID/*.inp /gpfs/scratch/$USER/$SLURM_JOB_ID/zip/
rsync -av /gpfs/scratch/$USER/$SLURM_JOB_ID/transfer/* /gpfs/scratch/$USER/$SLURM_JOB_ID/zip/transfer/

/bin/echo Simulation files in /gpfs/scratch/$USER/$SLURM_JOB_ID/zip.


# clean up and compression
rsync -av $SGE_O_WORKDIR/$JOB_NAME.o$SLURM_JOB_ID /gpfs/scratch/$USER/$SLURM_JOB_ID/
rsync -av $SGE_O_WORKDIR/$JOB_NAME.o$SLURM_JOB_ID /gpfs/scratch/$USER/$SLURM_JOB_ID/zip/

mkdir /data/SEMS-TaoLab/Niccolo-Forte/Ti/data/$DIS/$PATH_EXTRA/$fac/$LAT/$SLURM_JOB_ID/
mkdir /data/SEMS-TaoLab/Niccolo-Forte/Ti/data/$DIS/$PATH_EXTRA/$fac/$LAT/$SLURM_JOB_ID/zip/
rsync -av /gpfs/scratch/$USER/$SLURM_JOB_ID/zip/* /data/SEMS-TaoLab/Niccolo-Forte/Ti/data/$DIS/$PATH_EXTRA/$fac/$LAT/$SLURM_JOB_ID/zip/

if [ "$zip" = true ] ; then
    tar -czf C1_transfer-$LAT-$DIS-$SLURM_JOB_ID.tgz /gpfs/scratch/$USER/$SLURM_JOB_ID/transfer/
    tar -czf C2_zip-$LAT-$DIS-$SLURM_JOB_ID.tgz /gpfs/scratch/$USER/$SLURM_JOB_ID/zip/
    rsync -av /gpfs/scratch/$USER/$SLURM_JOB_ID/C1_transfer-$LAT-$DIS-$JOB_ID.tgz $SGE_O_WORKDIR
    rsync -av /gpfs/scratch/$USER/$SLURM_JOB_ID/C2_zip-$LAT-$DIS-$JOB_ID.tgz /data/SEMS-TaoLab/Niccolo-Forte/Ti/data/$DIS/$PATH_EXTRA/$fac/$LAT/
fi

if [ "$delete_scratch" = true ] ; then
    rm -rf /gpfs/scratch/$USER/$JOB_ID
fi

/bin/echo Job completed at: `date`.
/bin/echo Finished. Data saved in $SGE_O_WORKDIR and /data/SEMS-TaoLab/Niccolo-Forte/

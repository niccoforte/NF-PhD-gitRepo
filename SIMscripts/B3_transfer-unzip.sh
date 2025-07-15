echo LAT:
read LAT

# LAT=tri

cd Z:/p1/data/Ti/20disNodes/$LAT/

# rm -rf transfer

mkdir transfer
cd transfer

scp exy053@login.hpc.qmul.ac.uk:/data/SEMS-TaoLab/Niccolo-Forte/Ti/data/*/*/$LAT/*/zip/transfer/* "`pwd`"

mkdir Z:/p1/data/Ti/per/$LAT/transfer
cp *per* Z:/p1/data/Ti/per/$LAT/transfer

# tar -xvzf C*_$loc-$LAT-$DIS-$JOBn.tgz
# cp -r data/scratch/exy053/$JOBn/$loc/* "`pwd`"
# rm -rf data
# rm -rf C*_$loc-$LAT-$DIS-$JOBn.tgz
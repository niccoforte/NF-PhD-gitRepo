echo LAT:
read LAT

# LAT=tri

cd C:/Users/exy053
cd "OneDrive - Queen Mary, University of London"
cd Documents/Research/p1-LatticeFractureToughness/p1git-Lattices/code/data/Ti/20disNodes/$LAT/

# rm -rf transfer

mkdir transfer
cd transfer

scp exy053@login.hpc.qmul.ac.uk:/data/SEMS-TaoLab/Niccolo-Forte/Ti/data/*/*/$LAT/*/zip/transfer/* "`pwd`"

# tar -xvzf C*_$loc-$LAT-$DIS-$JOBn.tgz
# cp -r data/scratch/exy053/$JOBn/$loc/* "`pwd`"
# rm -rf data
# rm -rf C*_$loc-$LAT-$DIS-$JOBn.tgz
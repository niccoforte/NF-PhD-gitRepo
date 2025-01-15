echo LAT:
read LAT
echo nnx: 
read nnx
echo DIS:          # per, disNodes, disStruts
read DIS
echo JOBn:
read JOBn
echo loc:          # transfer, zip
read loc

# LAT=tri
# nnx=30
# DIS=disNodes
# JOBn=4664261
# loc=transfer

cd C:/Users/exy053
cd 'OneDrive - Queen Mary, University of London'
cd Documents/Research/Paper1-LatticeFractureToughness/p1git-Lattices/code/data/Ti/$LAT/

echo Running on host: `hostname`.
echo Starting on: `date`, in `pwd`.

if [[ $loc == "transfer" ]]; then
	scp exy053@login.hpc.qmul.ac.uk:/data/home/exy053/Paper1-LatticeFractureToughness/Ti/$LAT/C1_$loc-$LAT-$DIS-$JOBn.tgz "`pwd`"
elif [[ $loc == "zip" ]]; then
	scp exy053@login.hpc.qmul.ac.uk:/data/SEMS-TaoLab/Niccolo-Forte/Ti/data/C2_$loc-$LAT-$DIS-$JOBn.tgz "`pwd`"
fi

tar -xvzf C*_$loc-$LAT-$DIS-$JOBn.tgz

cp -r data/scratch/exy053/$JOBn/$loc/* "`pwd`"

rm -rf data
rm -rf C*_$loc-$LAT-$DIS-$JOBn.tgz

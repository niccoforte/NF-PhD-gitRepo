echo "LAT:"
read LAT

echo "Extra Path Spec (CHECK THIS)"
read EXTRA


#============ Transfer files from remote to local server ============

mkdir -p Z:/p1/data/Ti/disNodes/$EXTRA/0.2/$LAT/transfer
mkdir -p Z:/p1/data/Ti/per/$LAT/transfer
cd Z:/p1/data/Ti/disNodes/$EXTRA/0.2/$LAT/transfer

scp exy053@login.hpc.qmul.ac.uk:/data/SEMS-TaoLab/Niccolo-Forte/Ti/data/disNodes/$EXTRA/0.2/$LAT/*/zip/transfer/* exy053@login.hpc.qmul.ac.uk:/data/SEMS-TaoLab/Niccolo-Forte/Ti/data/per/0.0/$LAT/*/zip/transfer/* "`pwd`"

cp *per* Z:/p1/data/Ti/per/$LAT/transfer


# #============ Transfer files from local to remote server ============

# LOCAL_DIR="Z:/p1/sims/Ti/FrequencyDisorder/$LAT"
# REMOTE_DIR="/data/SEMS-TaoLab/Niccolo-Forte/Ti/data/disNodes/Frequency/0.2/$LAT/local/zip"

# cd $LOCAL_DIR

# ssh exy053@login.hpc.qmul.ac.uk "mkdir -p $REMOTE_DIR"
# # scp -r *.inp *.odb transfer exy053@login.hpc.qmul.ac.uk:"$REMOTE_DIR"
# scp -r transfer exy053@login.hpc.qmul.ac.uk:"$REMOTE_DIR"


# #============ Unzip files ============

# tar -xvzf C*_$loc-$LAT-$DIS-$JOBn.tgz
# cp -r data/scratch/exy053/$JOBn/$loc/* "`pwd`"
# rm -rf data
# rm -rf C*_$loc-$LAT-$DIS-$JOBn.tgz 
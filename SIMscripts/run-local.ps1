$LAT="tri"
$DIS="per"
$nnx=20
$unitCellSize=10
$mode="ductile"
$material="al"
$rD=0.2
$initial=1
$nJobs=1
$CPUs=12
$Fout=100
$Hout=200
$dir="al"

$strainApp=1

chdir C:\temp

abaqus cae noGUI=C:\Users\exy053\Documents\A1_FractureToughness-Ductility.py -- $LAT $DIS $nnx $unitCellSize $mode $material $rD $initial $nJobs $CPUs $Fout $Hout $dir
abaqus cae noGUI=C:\Users\exy053\Documents\A2_OUTpostProcess.py -- $LAT $DIS $nnx $unitCellSize $mode $material $rD $initial $nJobs $CPUs $Fout $Hout $dir
python C:\Users\exy053\Documents\A2_INpostProcess.py -- $LAT $DIS $nnx $unitCellSize $mode $material $rD $initial $nJobs $CPUs $Fout $Hout $dir

chdir C:\Users\exy053\Documents
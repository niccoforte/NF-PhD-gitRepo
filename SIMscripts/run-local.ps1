$LAT="tri"
$DIS="disNodes"
$nnx=30
$unitCellSize=10
$mode="ductile"
$material="ti"
$rD=0.2
$initial=1
$nJobs=100
$CPUs=12
$Fout=100
$Hout=200
$dir="Z:\\p1\sims\\Ti\\TargetedDisorder\\xs\\tri"

Set-Location C:\temp

for ($i = 1; $i -le 101; $i += 100) {
	$initial=$i
	abaqus cae noGUI=C:\Users\exy053\Documents\p1git-Lattices\SIMscripts\A1_FractureToughness-Ductility.py -- $LAT $DIS $nnx $unitCellSize $mode $material $rD $initial $nJobs $CPUs $Fout $Hout $dir
}

abaqus cae noGUI=C:\Users\exy053\Documents\p1git-Lattices\SIMscripts\A2_INpostProcess.py -- $LAT $DIS $nnx $unitCellSize $mode $material $rD $initial $nJobs $CPUs $Fout $Hout $dir
abaqus cae noGUI=C:\Users\exy053\Documents\p1git-Lattices\SIMscripts\A2_OUTpostProcess.py -- $LAT $DIS $nnx $unitCellSize $mode $material $rD $initial $nJobs $CPUs $Fout $Hout $dir

Set-Location C:\Users\exy053\Documents\p1git-Lattices\SIMscripts\

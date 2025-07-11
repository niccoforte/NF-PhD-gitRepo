$LAT="tri"
$nnx=30
$unitCellSize=10
$mode="ductile"
$material="ti"
$rD=0.2
$DIS="disNodes"
$fac=0.2
$distribution="frequency"
$target="all"
$initial=1
$nJobs=100
$CPUs=12
$Fout=100
$Hout=200
$dir="Z:\\p1\sims\\Ti\\FrequencyDisorder\\FCC"

Set-Location C:\temp

for ($i = 1; $i -le 401; $i += 100) {
	$initial=$i
	abaqus cae noGUI=C:\Users\exy053\Documents\p1git-Lattices\SIMscripts\A1_FractureToughness-Ductility.py -- $LAT $nnx $unitCellSize $mode $material $rD $DIS $fac $distribution $target $initial $nJobs $CPUs $Fout $Hout $dir
}

abaqus cae noGUI=C:\Users\exy053\Documents\p1git-Lattices\SIMscripts\A2_INpostProcess.py -- $LAT $nnx $unitCellSize $mode $material $rD $DIS $fac $distribution $target $initial $nJobs $CPUs $Fout $Hout $dir
abaqus cae noGUI=C:\Users\exy053\Documents\p1git-Lattices\SIMscripts\A2_OUTpostProcess.py -- $LAT $nnx $unitCellSize $mode $material $rD $DIS $fac $distribution $target $initial $nJobs $CPUs $Fout $Hout $dir

Set-Location C:\Users\exy053\Documents\p1git-Lattices\SIMscripts\

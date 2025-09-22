$LAT="FCC"
$nnx=16
$unitCellSize=10
$mode="ductile"
$material="ti"
$rD=0.2
$DIS="disNodes"
$fac=0.2
$distribution="opt"
$target="xs"
$initial=1
$nJobs=1
$CPUs=12
$Fout=100
$Hout=200
$dir="Z:\\p1\\data\\Ti\\disNodes\\Target-xs\\0.2\\FCC\\Opt"

Set-Location C:\temp

# for ($i = 1; $i -le 101; $i += 100) {
# 	$initial=$i
abaqus cae noGUI=C:\Users\exy053\Documents\p1git-Lattices\SIMscripts\A1_FractureToughness-Ductility.py -- $LAT $nnx $unitCellSize $mode $material $rD $DIS $fac $distribution $target $initial $nJobs $CPUs $Fout $Hout $dir "OptLoop" "1"
# }

abaqus cae noGUI=C:\Users\exy053\Documents\p1git-Lattices\SIMscripts\A2_INpostProcess.py -- $LAT $nnx $unitCellSize $mode $material $rD $DIS $fac $distribution $target $initial $nJobs $CPUs $Fout $Hout $dir
abaqus cae noGUI=C:\Users\exy053\Documents\p1git-Lattices\SIMscripts\A2_OUTpostProcess.py -- $LAT $nnx $unitCellSize $mode $material $rD $DIS $fac $distribution $target $initial $nJobs $CPUs $Fout $Hout $dir

Set-Location C:\Users\exy053\Documents\p1git-Lattices\SIMscripts\

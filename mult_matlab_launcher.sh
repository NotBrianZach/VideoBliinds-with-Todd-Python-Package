#!/bin/bash
x="a"
for j in `seq 1 20`;
do
    echo $x 
    y=$x
    x=$(echo "$x" | tr "0-9a-z" "1-9a-z_")
    sbatch matlablauncher.slurm
    sed -i "s/matlab_param_lista${y}/matlab_param_lista${x}/" matlablauncher.slurm
    echo "matlab_param_lista${y}"
    echo "matlab_param_lista${x}"
done  
sed -i "s/matlab_param_lista${x}/matlab_param_listaa/" matlablauncher.slurm


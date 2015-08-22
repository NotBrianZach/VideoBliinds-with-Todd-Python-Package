#!/bin/bash
x="a"
for j in `seq 1 20`;
do
    echo $x 
    y=$x
    x=$(echo "$x" | tr "0-9a-z" "1-9a-z_")
    sbatch movielauncher.slurm
    sed -i "s/movie_param_lista${y}/movie_param_lista${x}/" movielauncher.slurm
    echo "movie_param_lista${y}"
    echo "movie_param_lista${x}"
done  
sed -i "s/movie_param_lista${x}/movie_param_listaa/" movielauncher.slurm


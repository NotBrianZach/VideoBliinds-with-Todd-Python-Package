#!/bin/bash
inputDirs=~/frames_*;
j=0
for i in $inputDirs;
do
    Matlab -nojvm -nodesktop -r " arg1='${i}';run ./matLoaderTestMovie.m;run ./video_bliinds_algo.m;exit;"
    #need to run this from the directory of the python script to access other files needed to run.
    python ./vblind_scores.py ${i}
done

#!/bin/bash
#inputDirs=/scratch/01891/adb/ref_video_test/frames_*;
inputDirs=$1
echo $1
#inputDirs=~/frames_*;







#j=0
for i in $inputDirs;
do
    matlab -nojvm -nodesktop -r " arg1='${i}';run ./matLoaderTestMovie.m;run ./video_bliinds_algo.m;exit;"
    #need to run this from the directory of the python script to access other files needed to run.
#    python ./vblind_scores.py ${i}
done

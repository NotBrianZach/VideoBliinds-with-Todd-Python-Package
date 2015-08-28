#!/bin/bash
#inputDirs=/scratch/01891/adb/ref_video_test/frames_*;
inputDir=$1
vidName=`basename $1`
uniqueNum=$2
echo $1
#inputDirs=~/frames_*;

#j=0
#for i in $inputDirs;
#do
matlab -nojvm -nodesktop -r "arg1='${inputDir}';arg2='${vidName}';run ./matLoaderTestMovie.m;run ./video_bliinds_algo.m;exit;"
    #need to run this from the directory of the python script to access other files needed to run.
#python ./vblind_scores.py ${inputDir}  this produces a videobliinds score.
#    python ./vbliind_html_out.py
#done

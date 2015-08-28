#mon_dirs=(/scratch/01891/adb/Experiment1_Blanton/*.process/)
#mon_dirs=(/scratch/01891/adb/ref_video_test/*.process/)
#mon_dirs=($SCRATCH/vidFrames/*.process/)
mon_dirs=(/scratch/01891/adb/ref_video_test/*.process/)
mon_dirs=(${mon_dirs[@]})
#output_dir=$SCRATCH/movieRound2
for ((j=0;j<${#mon_dirs[@]};j++));
do 
    echo "./runMatlab.sh ${mon_dirs[$j]} ${j}" >> matlab_param_list
done

source activate mmlab
#conda activate mmlab

#######################################
### Change to your mmdetection path ###
#######################################
/path/to/mmdetection/tools/dist_train.sh  ./kuzu_htc_x101_64x4d_fpn_2gpu.py 2 \
    --work_dir ./work_dir 
    # --validate 
    #--resume_from ./work_dir/latest.pth


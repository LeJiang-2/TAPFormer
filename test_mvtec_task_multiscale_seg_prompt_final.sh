
device=2

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## test on the mvtec dataset
        base_dir=task_multiscale_final_train_on_visa
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test_task_multiscale_seg_prompt.py --dataset mvtec \
        --data_path /raid/jl/AnomalyGPT/data/mvtec_anomaly_detection --save_path ./results/${base_dir}/zero_shot_mvtec \
        --checkpoint_path ${save_dir} --epoch 1\
         --features_list 6 12 18 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
    wait
    done
done


device=1

LOG=${save_dir}"res.log"
echo ${LOG}
depth=(9)
n_ctx=(12)
t_n_ctx=(4)
for i in "${!depth[@]}";do
    for j in "${!n_ctx[@]}";do
    ## test on the VisA dataset
        base_dir=task_multiscale_final_train_on_mvtec
        save_dir=./checkpoints/${base_dir}/
        CUDA_VISIBLE_DEVICES=${device} python test_task_multiscale_seg_prompt.py --dataset visa \
        --data_path /raid/jl/AnomalyGPT/data/VisA --save_path ./results/${base_dir}/zero_shot_visa \
        --checkpoint_path ${save_dir} --epoch 1\
        --features_list 6 12 18 24 --image_size 518 --depth ${depth[i]} --n_ctx ${n_ctx[j]} --t_n_ctx ${t_n_ctx[0]}
    wait
    done
done
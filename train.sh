set -eu

#bash -x ./env.sh
DD_RAND_SEED=$1
source ./slurm/env.sh
source ./slurm/utils.sh
source ./model_conf

export PATH="$PWD/python/bin/:$PATH"
export PYTHONPATH="$PWD/python/"

export FLAGS_eager_delete_tensor_gb=0.0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_sync_nccl_allreduce=1
export NCCL_DEBUG=INFO
export NCCL_IB_GID_INDEX=3


e_executor=$(echo ${use_experimental_executor-'True'} | tr '[A-Z]' '[a-z]')

use_fuse=$(echo ${use_fuse-'False'} | tr '[A-Z]' '[a-z]')
if [[ ${use_fuse} == "true" ]]; then
    export FLAGS_fuse_parameter_memory_size=131072
    export FLAGS_fuse_parameter_groups_size=10
fi

#pack output
nohup sh ./slurm/pack_model.sh ./output > log/pack_model.log 2>&1 &

# check
#export iplist="10.255.137.19,10.255.120.23,10.255.84.15,10.255.105.20,10.255.134.16,10.255.73.15"
#export iplist="10.255.137.19,10.255.74.11,10.255.120.23,10.255.77.17,10.255.92.21,10.255.139.14,10.255.134.16,10.255.78.20"
#export iplist="10.255.74.11,10.255.91.19,10.255.120.23,10.255.122.15,10.255.92.21,10.255.139.14,10.255.134.16,10.255.66.17"
export iplist=`hostname -i`
check_iplist

distributed_args="--node_ips ${PADDLE_TRAINERS} --node_id ${PADDLE_TRAINER_ID} --current_node_ip ${POD_IP}"
python -u ./lanch.py ${distributed_args} \
    ./train.py --use_cuda "True" \
                --is_distributed "True" \
                --weight_sharing "True" \
                --use_fast_executor ${e_executor-"True"} \
                --use_fuse ${use_fuse-"False"} \
                --nccl_comm_num ${nccl_comm_num:-"1"} \
                --use_hierarchical_allreduce ${use_hierarchical_allreduce:-"False"} \
                --in_tokens "True" \
                --batch_size ${BATCH_SIZE} \
                --vocab_path ${vocab_path} \
                --task_group_json ${task_group_json} \
                --hack_old_data ${hack_old_data-"False"} \
                --generate_neg_sample ${generate_neg_sample-"True"} \
                --lr_scheduler ${lr_scheduler} \
                --num_train_steps ${num_train_steps} \
                --checkpoints ./output \
                --use_amp ${use_fp16:-"False"} \
                --use_recompute ${use_recompute:-"True"} \
                --use_dynamic_loss_scaling ${use_fp16} \
                --init_loss_scaling ${loss_scaling:-128} \
                --save_steps ${SAVE_STEPS} \
                --init_checkpoint ${init_model:-""} \
                --ernie_config_path ${CONFIG_PATH} \
                --learning_rate ${LR_RATE} \
                --warmup_steps ${WARMUP_STEPS:-0} \
                --weight_decay ${WEIGHT_DECAY:-0} \
                --max_seq_len ${MAX_LEN} \
                --skip_steps 10 >> log/job.log.${PADDLE_TRAINER_ID} 2>&1

set -eu

export FLAGS_eager_delete_tensor_gb=0
export FLAGS_fraction_of_gpu_memory_to_use=0.98
export FLAGS_sync_nccl_allreduce=1

e_executor=$(echo ${use_experimental_executor-'True'} | tr '[A-Z]' '[a-z]')
use_recompute=0
BATCH_SIZE=128
vocab_path="./config/vocab.txt"
generate_neg_sample="False"
lr_scheduler="linear_warmup_decay"
num_train_steps=500000
use_fp16="False"
loss_scaling=128000
SAVE_STEPS=10000
VALIDATION_STEPS=2
CONFIG_PATH="./config/ernie_config.json"
task_group_json="./config/task.json"
LR_RATE=1e-4
WARMUP_STEPS=0
WEIGHT_DECAY=0.01
MAX_LEN=64
use_experimental_executor="False"

#pack output
nohup sh ./slurm/pack_model.sh ./output > log/pack_model.log 2>&1 &

python3 -u \
    ./train.py --use_cuda "True" \
                --is_distributed "False" \
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
                --validation_steps ${VALIDATION_STEPS} \
                --warmup_steps ${WARMUP_STEPS:-0} \
                --weight_decay ${WEIGHT_DECAY:-0} \
                --max_seq_len ${MAX_LEN} \
                --skip_steps 10 >> log/job.log.0 2>&1

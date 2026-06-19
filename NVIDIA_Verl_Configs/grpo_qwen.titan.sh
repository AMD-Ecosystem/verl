#!/bin/bash

# MODEL SPECIFIC PATHS
MODEL_PATH="Qwen/Qwen2-7B-Instruct"
train_files="../data/gsm8k/train.parquet"
test_files="../data/gsm8k/test.parquet"

# DEFAULT VALUES
export TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1024}
export MINI_BATCH_SIZE=${MINI_BATCH_SIZE:-256}
export MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-80}
export TP_VALUE=${TP_VALUE:-2}
export INFERENCE_BATCH_SIZE=${INFERENCE_BATCH_SIZE:-40}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.4}
export ROLLOUT_N=${ROLLOUT_N:-5}
export EPOCHS=${EPOCHS:-50}

# TRITON CUDA COMPILER FIX
export TRITON_PTXAS_PATH=$(which ptxas)

# CONFIGURATION LOG
echo "========================================================="
echo " PERFORMANCE CONFIGURATION (GRPO)"
echo " Run Date:                $(date '+%Y-%m-%d %H:%M:%S')"
echo " Model:                   $MODEL_PATH"
echo " EPOCHS:                  $EPOCHS"
echo " TP Value:                $TP_VALUE"
echo " Rollout Group Size:      $ROLLOUT_N"
echo " TRAIN_BATCH_SIZE:        $TRAIN_BATCH_SIZE"
echo " MINI_BATCH_SIZE:         $MINI_BATCH_SIZE"
echo " MICRO_BATCH_SIZE:        $MICRO_BATCH_SIZE"
echo " INFERENCE_BATCH_SIZE:    $INFERENCE_BATCH_SIZE"
echo "========================================================="

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export WANDB_API_KEY="eb0e6fdfaffbae8ec1535bf05ee6aebe298796ea"
export TORCH_NCCL_USE_TENSOR_REGISTER_ALLOCATOR_HOOK=0
export VLLM_ALLREDUCE_USE_SYMM_MEM=0
export VLLM_USE_NCCL_SYMM_MEM=0
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:False"
GPUS_PER_NODE=8

python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$train_files \
        data.val_files=$test_files \
        data.train_batch_size=$TRAIN_BATCH_SIZE \
        data.max_prompt_length=512 \
        data.max_response_length=1024 \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=$MINI_BATCH_SIZE \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.001 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_VALUE \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
        actor_rollout_ref.rollout.n=$ROLLOUT_N \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console'] \
        trainer.project_name='grpo_qwen_llm' \
        trainer.experiment_name='grpo_trainer/run_qwen2-7b.sh_default' \
        trainer.n_gpus_per_node=$GPUS_PER_NODE \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=10 \
        trainer.total_epochs=$EPOCHS \
        2>&1 | tee /dev/stderr | grep -oP 'perf/throughput:\K\d+\.\d+' | \
        awk 'NR > 2 {sum += 1/$1; count++} END {if (count > 0) print "\n>> Harmonic Mean: " count/sum " tokens/sec"}'

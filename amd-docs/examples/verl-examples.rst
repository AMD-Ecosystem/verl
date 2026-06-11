.. meta::
  :description: verl examples
  :keywords: verl, programming, ROCm, example, sample, tutorial

.. _run-a-verl-example:

********************************************************************
Run a verl example
********************************************************************

This guide shows how to run a verl example on AMD GPUs with ROCm.
It covers data preparation, model loading checks, configuration,
environment variables, running PPO and GRPO, and launching multi-node
training with Slurm.

1. Prepare the GSM8K dataset using the provided preprocessing script from the examples directory at `https://github.com/ROCm/verl/tree/main/examples/data_preprocess <https://github.com/ROCm/verl/tree/main/examples/data_preprocess>`__.

   .. code-block:: bash

      python3 examples/data_preprocess/gsm8k.py --local_dir ../data/gsm8k

2. Verify that the selected models can be loaded with Hugging Face Transformers.

   .. code-block:: bash

      python3 -c "import transformers; transformers.pipeline('text-generation', model='Qwen/Qwen2-7B-Instruct')"
      python3 -c "import transformers; transformers.pipeline('text-generation', model='deepseek-ai/deepseek-llm-7b-chat')"

3. Set the model path and dataset files, and choose any model supported by verl for ``MODEL_PATH``; this example uses ``Qwen/Qwen2-7B-Instruct`` and ``deepseek-ai/deepseek-llm-7b-chat``, and uses GSM8K formatted with the provided preprocessing code.

   .. code-block:: bash

      MODEL_PATH="Qwen/Qwen2-7B-Instruct"
      train_files="../data/gsm8k/train.parquet"
      test_files="../data/gsm8k/test.parquet"

4. Set the ROCm environment variables and the number of GPUs per node; you must set ``HIP_VISIBLE_DEVICES`` and ``ROCR_VISIBLE_DEVICES``, which is the key difference compared to running with PyTorch on CUDA.

   .. code-block:: bash

      export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
      export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
      GPUS_PER_NODE=8

5. Run PPO by defining the parameters and launching training.

   .. note::
      If you use ``deepseek-ai/deepseek-llm-7b-chat``, set ``TP_VALUE=4``, ``INFERENCE_BATCH_SIZE=32``, and ``GPU_MEMORY_UTILIZATION=0.4``.

   .. code-block:: bash

      MODEL_PATH="Qwen/Qwen2-7B-Instruct"  # You can use: deepseek-ai/deepseek-llm-7b-chat
      TP_VALUE=2
      INFERENCE_BATCH_SIZE=32
      GPU_MEMORY_UTILIZATION=0.4

      python3 -m verl.trainer.main_ppo  \
        data.train_files=$train_files  \
        data.val_files=$test_files  \
        data.train_batch_size=1024 \
        data.max_prompt_length=1024 \
        data.max_response_length=512 \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
        actor_rollout_ref.rollout.tensor_model_parallel_size=$TP_VALUE \
        actor_rollout_ref.rollout.name=vllm  \
        actor_rollout_ref.rollout.gpu_memory_utilization=$GPU_MEMORY_UTILIZATION \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        critic.optim.lr=1e-5 \
        critic.model.use_remove_padding=True \
        critic.model.path=$MODEL_PATH \
        critic.model.enable_gradient_checkpointing=True \
        critic.ppo_micro_batch_size_per_gpu=32 \
        critic.model.fsdp_config.param_offload=False \
        critic.model.fsdp_config.optimizer_offload=False \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name='ppo_qwen_llm' \
        trainer.experiment_name='ppo_trainer/run_qwen2-7b.sh_default' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=10 \
        trainer.total_epochs=50

6. Run GRPO by setting the advantage estimator and launching training.

   .. note::
      If you use ``deepseek-ai/deepseek-llm-7b-chat``, set ``TP_VALUE=2``, ``INFERENCE_BATCH_SIZE=110``, and ``GPU_MEMORY_UTILIZATION=0.6``.

   .. code-block:: bash

      MODEL_PATH="Qwen/Qwen2-7B-Instruct"  # You can use: deepseek-ai/deepseek-llm-7b-chat
      TP_VALUE=2
      INFERENCE_BATCH_SIZE=40
      GPU_MEMORY_UTILIZATION=0.6

      python3 -m verl.trainer.main_ppo \
        algorithm.adv_estimator=grpo \
        data.train_files=$train_files \
        data.val_files=$test_files \
        data.train_batch_size=1024 \
        data.max_prompt_length=512 \
        data.max_response_length=1024 \
        actor_rollout_ref.model.path=$MODEL_PATH \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.model.use_remove_padding=True \
        actor_rollout_ref.actor.ppo_mini_batch_size=256 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=80 \
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
        actor_rollout_ref.rollout.n=5 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$INFERENCE_BATCH_SIZE \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        trainer.critic_warmup=0 \
        trainer.logger=['console','wandb'] \
        trainer.project_name='grpo_qwen_llm' \
        trainer.experiment_name='grpo_trainer/run_qwen2-7b.sh_default' \
        trainer.n_gpus_per_node=8 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=10 \
        trainer.total_epochs=50

7. Launch multi-node training with Slurm after you finish single-node training on AMD clusters that use Slurm.

   .. code-block:: bash

      sbatch slurm_script.sh

   For more detailed guidance, see the single-node and multi-node training
   tutorials in the official `verl <https://verl.readthedocs.io/en/latest/>`__
   upstream documentation.
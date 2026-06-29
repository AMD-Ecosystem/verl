.. meta::
  :description: verl fully asynchronous examples
  :keywords: verl, programming, ROCm, example, DAPO, GRPO, Megatron, FSDP2

.. _run-a-fully-async-verl-example:

********************************************************************
Run fully asynchronous verl examples
********************************************************************

This guide shows how to run fully asynchronous verl examples on AMD GPUs with ROCm. It covers data and model preparation, launching fully asynchronous GRPO training on a vision-language model with Megatron and fully asynchronous DAPO math reasoning training with FSDP2.

Megatron example
--------------------

The `geo3k_qwen25vl_7b_megatron_4_4.sh <https://github.com/ROCm/verl/blob/main/verl/experimental/fully_async_policy/shell/geo3k_qwen25vl_7b_megatron_4_4.sh>`_ example launches fully asynchronous GRPO training for ``Qwen2.5-VL-7B-Instruct`` on the Geometry3k vision-math dataset using verl's fully asynchronous policy with the Megatron trainer configuration.

1. Download `prepare_geo3k_qwen25vl_7b_megatron_4_4.sh <https://github.com/ROCm/verl/tree/amd-integration/verl/experimental/fully_async_policy/shell/data_model_preparation/prepare_geo3k_qwen25vl_7b_megatron_4_4.sh>`_ and run:

   .. code-block:: bash

      export HF_TOKEN=your_token
      # or: hf auth login
      cd /workspace/verl
      bash verl/experimental/fully_async_policy/shell/data_model_preparation/prepare_geo3k_qwen25vl_7b_megatron_4_4.sh

2. Run the example:

   .. code-block:: bash

      export HF_MODEL_PATH=${HOME}/models/Qwen2.5-VL-7B-Instruct
      cd /workspace/verl
      bash verl/experimental/fully_async_policy/shell/geo3k_qwen25vl_7b_megatron_4_4.sh

   This example will take several hours to run. Once it has completed, its output should be similar to this output:

   .. image:: ../data/geo3k_qwen25vl_7b_megatron_4_4_complete.png
      :alt: Expected terminal output after geo3k_qwen25vl_7b_megatron_4_4.sh completes

DAPO example
--------------------------------------------------

The `dapo_7b_math_fsdp2_4_4.sh <https://github.com/ROCm/verl/blob/main/verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_4_4.sh>`__
example launches fully asynchronous DAPO reinforcement learning training using ``Qwen2.5-Math-7B`` on math reasoning tasks.

1. Download `prepare_dapo_7b_math_fsdp2_4_4.sh <https://github.com/ROCm/verl/tree/amd-integration/verl/experimental/fully_async_policy/shell/data_model_preparation/prepare_dapo_7b_math_fsdp2_4_4.sh>`_ and run:

   .. code-block:: bash

      export HF_TOKEN=your_token
      # or: hf auth login
      cd /workspace/verl
      bash verl/experimental/fully_async_policy/shell/data_model_preparation/prepare_dapo_7b_math_fsdp2_4_4.sh

3. Run the example:

   .. code-block:: bash

      cd /workspace/verl
      bash verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_4_4.sh

   This example will take several hours to run. Once it has completed, its output should be similar to this output:

   .. image:: ../data/dapo_7b_math_fsdp2_4_4_complete.png
      :alt: Expected terminal output after dapo_7b_math_fsdp2_4_4.sh completes

4. If you encounter out of memory issues, reduce ``max_position_embeddings`` in the ``config.json`` to 4096 and apply these changes in
   ``dapo_7b_math_fsdp2_4_4.sh``:

   .. code-block:: bash

      python -c "
      import json, pathlib
      config_path = pathlib.Path.home() / 'verl/models/Qwen2.5-Math-7B/config.json'
      with open(config_path) as f:
          config = json.load(f)
      config['max_position_embeddings'] = 4096
      with open(config_path, 'w') as f:
          json.dump(config, f, indent=2)
      "

      # line 55: default uses * 2
      actor_ppo_max_token_len=$((max_prompt_length + max_response_length))

      # line 56: default uses * 3
      infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))

      # line 61: default is fsdp_size=2
      fsdp_size=4

      # line 74: default is total_rollout_steps=$(((512*100)))
      total_rollout_steps=$((100))

      # line 105: default is max_position_embeddings=32768
      +actor_rollout_ref.model.override_config.max_position_embeddings=4096 \

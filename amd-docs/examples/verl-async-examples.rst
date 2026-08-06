.. meta::
  :description: verl fully asynchronous examples
  :keywords: verl, programming, ROCm, example, DAPO, GRPO, Megatron, FSDP2

.. _run-a-fully-async-verl-example:

********************************************************************
Run fully asynchronous verl examples
********************************************************************

This guide shows how to run fully asynchronous verl examples on AMD GPUs with ROCm.
It covers preparing data and models, launching fully asynchronous GRPO training on a
vision-language model with Megatron, and running fully asynchronous DAPO math
reasoning training with FSDP2.

Megatron example
--------------------------------------------------------------------

The `geo3k_qwen25vl_7b_megatron_4_4.sh <https://github.com/AMD-Ecosystem/verl/blob/main/verl/experimental/fully_async_policy/shell/geo3k_qwen25vl_7b_megatron_4_4.sh>`_ example launches fully asynchronous GRPO training for ``Qwen2.5-VL-7B-Instruct`` on the Geometry3k vision-math dataset using verl's fully asynchronous policy with the Megatron trainer configuration.

1. Download `prepare_geo3k_qwen25vl_7b_megatron_4_4.sh <https://github.com/AMD-Ecosystem/verl/tree/amd-integration/verl/experimental/fully_async_policy/shell/data_model_preparation/prepare_geo3k_qwen25vl_7b_megatron_4_4.sh>`_ and run:

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

   This example will take several hours to run. Once the example completes, the output should resemble the following:

   .. image:: ../data/geo3k_qwen25vl_7b_megatron_4_4_complete.png
      :alt: Expected terminal output after geo3k_qwen25vl_7b_megatron_4_4.sh completes

DAPO example
--------------------------------------------------------------------

The `dapo_7b_math_fsdp2_4_4.sh <https://github.com/AMD-Ecosystem/verl/blob/main/verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_4_4.sh>`__
example launches fully asynchronous DAPO reinforcement learning training using ``Qwen2.5-Math-7B`` on math reasoning tasks.

1. Download `prepare_dapo_7b_math_fsdp2_4_4.sh <https://github.com/AMD-Ecosystem/verl/tree/amd-integration/verl/experimental/fully_async_policy/shell/data_model_preparation/prepare_dapo_7b_math_fsdp2_4_4.sh>`_ and run:

   .. code-block:: bash

      export HF_TOKEN=your_token
      # or: hf auth login
      cd /workspace/verl
      bash verl/experimental/fully_async_policy/shell/data_model_preparation/prepare_dapo_7b_math_fsdp2_4_4.sh

2. Run the example:

   .. code-block:: bash

      cd /workspace/verl
      bash verl/experimental/fully_async_policy/shell/dapo_7b_math_fsdp2_4_4.sh

   This example will take several hours to run. Once the example completes, the output should resemble the following:

   .. image:: ../data/dapo_7b_math_fsdp2_4_4_complete.png
      :alt: Expected terminal output after dapo_7b_math_fsdp2_4_4.sh completes



.. meta::
  :description: What is verl?
  :keywords: verl, documentation, vLLM, reinforcement learning, deep learning, framework, GPU, AMD, ROCm, overview, introduction

.. _what-is-verl:

********************************************************************
What is verl?
********************************************************************

Volcano Engine Reinforcement Learning for LLMs (`verl <https://verl.readthedocs.io/en/latest/>`__)
is a reinforcement learning (RL) training library designed for the
post-training of large language models. It is the open-source version
of `HybridFlow <https://arxiv.org/abs/2409.19256v2>`__, which models
post-training workflows as a dataflow graph.

It provides a flexible and scalable system for implementing Reinforcement
Learning from Human Feedback (RLHF) and other RL-based optimization
workflows. The library integrates with modern language model training
and inference stacks while prioritizing performance and modularity.

verl is part of the `ROCm-LLMExt toolkit <https://rocm.docs.amd.com/projects/rocm-llmext/en/docs-26.02/>`__.

Why verl?
====================================================================

verl is well suited for RL because:

- Its **hybrid programming model** reduces complexity in RL dataflow
  construction while maintaining flexibility for a variety of algorithms.

- The **modular APIs** support reuse and extension of existing
  infrastructure and model ecosystems, reducing engineering overhead.

- **Performance and scalability** are core design goals, supporting
  efficient resource use across GPU clusters and multi-node,
  multi-framework training scenarios.

- Active community engagement and open-source development make it suitable
  for both research and production workflows.

verl features and use cases
====================================================================

verl provides the following features:

- **Flexible RL Algorithms:** Supports extension and implementation
  of diverse RL algorithms using a hybrid programming model that unifies
  single-controller and multi-controller paradigms for efficient dataflow
  execution with minimal code.

- **Modular Integration:** Integrates with existing LLM infrastructure
  such as PyTorch FSDP, FSDP2, Megatron-LM, vLLM, SGLang, and Hugging
  Face models, decoupling computation and data dependencies.

- **Scalable Parallelism:** Flexible device mapping and parallelism
  support efficient use of multi-GPU and distributed cluster environments.

- **High Performance:** Achieves training and rollout throughput through
  tight integration with optimized engines, and uses techniques such
  as efficient actor model resharding through the 3D-HybridEngine to
  reduce memory and communication overhead.

- **Models and configuration:** Supports common model families such as
  Qwen, Llama, Gemma, and DeepSeek via Hugging Face, with YAML-based
  configuration and example scripts for datasets such as GSM8K.

verl is commonly used in the following scenarios:

- **RLHF Training for LLMs:** Train language models with RL algorithms
  such as Proximal Policy Optimization (PPO) and Group Relative Policy
  Optimization (GRPO) for RLHF and other alignment recipes.

- **Agent Training:** Build RL-based agent training pipelines that
  interact with environments or tools.

- **Research and Experimentation:** Rapidly prototype and evaluate different
  RL strategies and configurations on large-scale models.

- **Production Deployments:** Integrate production-ready RL workflows
  using diverse backends and distributed computing resources.

ROCm deployment and runtime
====================================================================

On ROCm, verl uses the same HybridFlow controller and worker layout as
upstream, with AMD-specific container images and runtime.

- ``rocm/verl`` ships verl, PyTorch, ROCm, and vLLM for the default
  AMD rollout stack.

- Ray worker placement includes HIP device visibility handling for
  multi-GPU scheduling on ROCm clusters.

- Layouts span single-node jobs and multi-node clusters, including
  Slurm-managed Ray deployments.



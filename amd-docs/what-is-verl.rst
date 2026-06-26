.. meta::
  :description: What is verl?
  :keywords: verl, documentation, vLLM, reinforcement learning, deep learning, framework, GPU, AMD, ROCm, overview, introduction

.. _what-is-verl:

********************************************************************
What is verl?
********************************************************************

`verl <https://verl.readthedocs.io/en/latest/>`__ is a reinforcement
learning training library designed for post-training large language
models. It provides a flexible and scalable system for implementing
reinforcement learning from human feedback and other reinforcement
learning–based optimization workflows. The library is built to
integrate with modern language model training and inference stacks
while prioritizing performance, modularity, and ease of use.

Features and use cases
====================================================================

verl provides the following key features:

- **Flexible RL Algorithms:** Supports extension and implementation
  of diverse RL algorithms using a hybrid programming model that unifies
  single-controller and multi-controller paradigms for efficient data
  flow execution with minimal code.

- **Modular Integration:** Seamlessly integrates with existing LLM
  infrastructure such as PyTorch FSDP, Megatron-LM, vLLM, SGLang, and
  Hugging Face models, decoupling computation and data dependencies.

- **Scalable Parallelism:** Flexible device mapping and parallelism
  support allow efficient utilization of multi-GPU and distributed
  cluster environments.

- **High Performance:** Achieves training and rollout throughput through
  tight integration with optimized engines, and uses techniques such
  as efficient actor model resharding through the 3D-HybridEngine to
  reduce memory and communication overhead.

verl is commonly used in the following scenarios:

- **RLHF Training for LLMs:** Train language models with RLHF algorithms
  such as PPO, GRPO, and other recipes for improved alignment and quality.

- **Agent Training:** Build and scale RL-based agent training pipelines
  that interact with environments or tools at scale.

- **Research and Experimentation:** Rapidly prototype and evaluate different
  RL strategies and configurations on large-scale models.

- **Production Deployments:** Integrate production-ready RL workflows
  using diverse backends and distributed computing resources.

Why verl?
====================================================================

verl is well suited for those who require reinforcement learning for the following reasons:

- Its **hybrid programming model** simplifies complex RL dataflow
  construction while maintaining flexibility for a variety of algorithms.

- The **modular APIs** allow easy reuse and extension with existing
  infrastructure and model ecosystems, reducing engineering overhead.

- **Performance and scalability** are core design goals, enabling
  efficient resource use across GPU clusters and support for multi-node,
  multi-framework training scenarios.

- Active community engagement and open-source development make it suitable
  for both research and production workflows.

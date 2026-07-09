.. meta::
  :description: verl documentation
  :keywords: verl, ROCm, documentation, vLLM, reinforcement learning, deep learning, framework, GPU

.. _verl-documentation-index:

********************************************************************
verl on ROCm documentation
********************************************************************

Use verl on ROCm to train reinforcement learning loops for LLMs on AMD Instinct GPUs,
enabling efficient, scalable policy optimization for production copilots and autonomous agents.

Volcano Engine Reinforcement Learning for LLMs (`verl <https://verl.readthedocs.io/en/latest/>`__)  
is a reinforcement learning library designed for large language models (LLMs). 
verl offers a scalable, open-source fine-tuning solution by using a hybrid programming model 
that makes it easy to define and run complex post-training dataflows efficiently. 

Its modular APIs separate computation from data, allowing smooth integration with other frameworks. 
It also supports flexible model placement across GPUs for efficient scaling on different cluster sizes.
verl achieves high training and generation throughput by building on existing LLM frameworks. 
Its 3D-HybridEngine reduces memory use and communication overhead when switching between training 
and inference, improving overall performance.

verl is part of the `ROCm-LLMExt toolkit
<https://rocm.docs.amd.com/projects/rocm-llmext/en/docs-26.02/>`__.

The verl public repository is located at `https://github.com/ROCm/verl <https://github.com/ROCm/verl>`__.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install verl <install/verl-install>`

  .. grid-item-card:: Examples

    * :doc:`Run a verl example <examples/verl-examples>`

  .. grid-item-card:: Reference

      * `Quickstart PPO training on GSM8K dataset (upstream) <https://verl.readthedocs.io/en/latest/start/quickstart.html>`__
      * `HybridFlow programming guide (upstream) <https://verl.readthedocs.io/en/latest/hybrid_flow.html>`__

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the :doc:`Licensing <about/license>` page.

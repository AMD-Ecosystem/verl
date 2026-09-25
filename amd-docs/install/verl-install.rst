.. meta::
  :description: installing verl for ROCm
  :keywords: installation instructions, Docker, AMD, ROCm, verl

.. _verl-on-rocm-installation:

********************************************************************
verl on ROCm installation
********************************************************************

System requirements
====================================================================

To use verl `0.9.0 <https://github.com/volcengine/verl/releases/tag/v0.9.0>`__, you need the following prerequisites:

- **ROCm version:** `10.0.0 <https://rocm.docs.amd.com/en/docs-10.0.0/>`__
- **Operating system:** Ubuntu 24.04
- **GPU platform:** AMD Instinct™ MI300X, MI325X, MI350X and MI355X
- **PyTorch:** `2.12.0 <https://github.com/ROCm/pytorch/tree/release/2.12>`__
- **Python:** `3.12 <https://www.python.org/downloads/release/python-31213/>`__
- **vLLM:** `0.27.0 <https://github.com/vllm-project/vllm/releases/tag/v0.27.0>`__

Install verl
====================================================================

To run verl with ROCm enabled, build from source using the provided Dockerfile.

Build verl from source
--------------------------------------------------------------------

1. Clone the verl repository containing the Dockerfile:

   .. code-block:: bash

      git clone --recursive --branch release/0.9.0.amd0 https://github.com/AMD-Ecosystem/verl.git
      cd verl/docker/rocm

2. Build the Docker image:

   .. code-block:: bash

      docker build -t verl-release-v0.9.0amd0 -f Dockerfile.rocm .

   This builds an image with verl 0.9.0 and the required dependencies, including
   PyTorch 2.12.0, vLLM 0.27.0, SGLang 0.5.19, and Ray 2.58.0.

3. Run the Docker container:

   .. code-block:: bash

      mkdir -p $HOME/verl-workspace
      docker run -it \
         --name verl-release \
         --device /dev/kfd \
         --device /dev/dri \
         --privileged \
         --network=host \
         --group-add video \
         --cap-add=SYS_PTRACE \
         --security-opt seccomp=unconfined \
         --shm-size=2048g \
         --ulimit memlock=-1 \
         --ulimit stack=67108864 \
         -v $HOME/verl-workspace:/verl-workspace \
         -w /workspace \
         verl-release-v0.9.0amd0 \
         /bin/bash

   .. note::

      ``--shm-size=2048g`` is the recommended shared-memory allocation for
      large multi-GPU ROCm training in this image.

Test the verl installation
====================================================================

After starting the container, verify that verl and ROCm are working correctly.

1. Confirm that verl is installed:

   .. code-block:: bash

      pip list | grep verl

   Expected output:

   .. code-block:: text

      verl    0.9.0

2. Confirm GPU visibility from the container. On ROCm, PyTorch reports AMD GPUs
   through the CUDA-compatible API:

   .. code-block:: bash

      python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

   This should print ``True`` and the name of your AMD Instinct GPU (for example,
   ``AMD Instinct MI300X``, ``MI325X``,  ``MI350X`` or ``MI355X``).

3. Verify key dependency versions:

   .. code-block:: bash

      python -c "import torch; print('PyTorch', torch.__version__)"
      pip show vllm ray | grep -E '^(Name|Version):'

If all checks pass, proceed to run a verl example or run a PPO or GRPO training
workflow.

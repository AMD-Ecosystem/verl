.. meta::
  :description: installing verl for ROCm
  :keywords: installation instructions, Docker, AMD, ROCm, verl

.. _verl-on-rocm-installation:

********************************************************************
verl on ROCm installation
********************************************************************

System requirements
====================================================================

To use verl `0.6.0 <https://github.com/volcengine/verl/releases/tag/v0.6.0>`__, you need the following prerequisites:

- **ROCm version:** `7.0.0 <https://rocm.docs.amd.com/en/docs-7.0.0/>`__
- **Operating system:** Ubuntu 22.04
- **GPU platform:** AMD Instinct™ MI300X
- **PyTorch:** `2.9.0 <https://github.com/ROCm/pytorch/tree/release/2.9-rocm7.x-gfx115x>`__
- **Python:** `3.12 <https://www.python.org/downloads/release/python-31211/>`__
- **vLLM:** `0.11.0 <https://github.com/vllm-project/vllm/releases/tag/v0.11.0>`__

Install verl
================================================================================

To install verl on ROCm, you have the following options:

- :ref:`Use the prebuilt Docker image <use-docker-with-verl-pre-installed>` **(recommended)**
- :ref:`Build your own docker image <build-your-verl-rocm-docker-image>`

.. _use-docker-with-verl-pre-installed:

Use a prebuilt Docker image with verl pre-installed
--------------------------------------------------------------------------------

The recommended way to set up a verl environment and avoid potential installation issues is with Docker. 
The tested, prebuilt image includes verl, PyTorch, ROCm, and other dependencies.

Prebuilt Docker images with verl configured for ROCm are available on `Docker Hub <https://hub.docker.com/r/rocm/verl/tags>`_.

1. Pull the Docker image

   .. code-block:: bash

      docker pull rocm/verl:verl-0.6.0.amd0_rocm7.0_vllm0.11.0.dev

2. Launch and connect to the Docker container

   .. code-block:: bash

      docker run --rm -it \
         --name rocm_verl \
         --device /dev/dri \
         --device /dev/kfd \
         --group-add video \
         --cap-add SYS_PTRACE \
         --security-opt seccomp=unconfined \
         --privileged \
         -p 8265:8265 \
         -v "$HOME/.ssh:/root/.ssh" \
         -v "$HOME:$HOME" \
         --shm-size 128G \
         -w "$PWD" \
         rocm/verl:verl-0.6.0.amd0_rocm7.0_vllm0.11.0.dev \
         /bin/bash

.. _build-your-verl-rocm-docker-image:

Build your own Docker image
--------------------------------------------------------------------------------

1. Clone the `https://github.com/ROCm/verl <https://github.com/ROCm/verl>`_ repository

   .. code-block:: bash

      git clone https://github.com/ROCm/verl.git

2. Build the Docker container using the Dockerfile in the ``verl/docker`` directory

   .. code-block:: bash

      cd verl
      docker build --build-arg VLLM_REPO=https://github.com/vllm-project/vllm.git \
         --build-arg VLLM_BRANCH=4ca5cd5740c0cd7788cdfa8b7ec6a27335607a48 \
         --build-arg VERL_REPO=https://github.com/ROCm/verl.git \
         --build-arg VERL_BRANCH=0eb50ec4a33cda97e05ed8caab9c7f17a30c05a9 \
         -f docker/Dockerfile.rocm7 -t my-rocm-verl .

3. Launch and connect to the container

   .. code-block:: bash

      docker run --rm -it \
         --device /dev/dri \
         --device /dev/kfd \
         -p 8265:8265 \
         --group-add video \
         --cap-add SYS_PTRACE \
         --security-opt seccomp=unconfined \
         --privileged \
         -v $HOME/.ssh:/root/.ssh \
         -v $HOME:$HOME \
         --shm-size 128G \
         -w $PWD \
         --name rocm_verl \
         my-rocm-verl \
         /bin/bash

   .. note::

      The ``--shm-size`` parameter allocates shared memory for the container. It can be adjusted based on your system's resources.

Test the verl installation
================================================================================

Once connected to the Docker container, verify that verl is installed:

.. code-block:: bash 

   pip list | grep verl
   verl    0.3.0.post0        /app

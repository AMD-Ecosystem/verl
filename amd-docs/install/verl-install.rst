.. meta::
  :description: installing verl for ROCm
  :keywords: installation instructions, Docker, AMD, ROCm, verl

.. _verl-on-rocm-installation:

********************************************************************
verl on ROCm installation
********************************************************************

System requirements
====================================================================

To use verl `0.7.1 <https://github.com/volcengine/verl/releases/tag/v0.7.1>`__, you need the following prerequisites:

- **ROCm version:** `7.0.2 <https://rocm.docs.amd.com/en/docs-7.0.2/>`__
- **Operating system:** Ubuntu 22.04
- **GPU platform:** AMD Instinct™ MI300X, MI325X, MI355X
- **PyTorch:** `2.9.1 <https://github.com/ROCm/pytorch/tree/release/2.9-rocm7.x-gfx115x>`__
- **Python:** `3.12 <https://www.python.org/downloads/release/python-31211/>`__
- **vLLM:** `0.20.2 <https://github.com/vllm-project/vllm/releases/tag/v0.20.2>`__

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

      docker pull rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2

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
         rocm/verl:verl-0.7.1.amd0_rocm7.0.2_ubuntu22.04_py3.12_vllm0.20.2 \
         /bin/bash

.. _build-your-verl-rocm-docker-image:

Build your own Docker image
--------------------------------------------------------------------------------

1. Download the Dockerfile from the ``release/0.7.1.amd0`` branch:

   .. code-block:: bash

      curl -O https://raw.githubusercontent.com/AMD-Ecosystem/verl/release/0.7.1.amd0/docker/rocm/verl0.7.1-amd0/Dockerfile

2. Build the Docker image:

   .. code-block:: bash

      docker build -t verl-release-v0.7.1amd0 .

3. Run the Docker container:

   .. code-block:: bash

      docker run -it --name verl-release --device /dev/kfd --device /dev/dri \
         --privileged --network=host \
         --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
         --shm-size=2048g \
         --ulimit memlock=-1 --ulimit stack=67108864 \
         -w /workspace \
         verl-release-v0.7.1amd0 \
         /bin/bash

   .. note::

      The ``--shm-size`` parameter allocates shared memory for the container. It can be adjusted based on your system's resources.

Test the verl installation
================================================================================

Once connected to the Docker container, verify that verl is installed:

.. code-block:: bash 

   pip list | grep verl
   verl    0.7.1        /app

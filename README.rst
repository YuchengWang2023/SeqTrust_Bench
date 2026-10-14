SeqTrust_Bench
==============

A benchmarking framework for DNA sequence design algorithms

Installation
------------

Prerequisites
~~~~~~~~~~~~~

- `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or Anaconda
- A CUDA-capable GPU (the environment uses PyTorch 2.10.0 with CUDA 12.8)
- The prebuilt FlashAttention wheel (see below)

Step 0: Obtain the FlashAttention wheel
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

FlashAttention must match PyTorch 2.10 + CUDA 12 (``cxx11abi=TRUE``, Python 3.12).
The prebuilt wheel (~126 MB) is too large to host on GitHub, so it is **not
included in this repository**. Obtain the file
``flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl``
separately (e.g. from your group's shared storage), or build it from source.

One-click setup (recommended)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Pass the path of the wheel to ``setup_env.sh``:

.. code-block:: bash

   bash setup_env.sh /path/to/flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

The script will automatically:

1. Create the ``SeqTrust_Bench`` conda environment (Python 3.12.13)
2. Install all dependencies from ``requirements.txt``
3. Install FlashAttention 2.8.1 from the wheel you provided
4. Verify the installation

When it finishes, activate the environment with:

.. code-block:: bash

   conda activate SeqTrust_Bench

Manual setup
~~~~~~~~~~~~

If you prefer to install step by step:

.. code-block:: bash

   # 1. Create and activate the conda environment
   conda env create -f environment.yml
   conda activate SeqTrust_Bench

   # 2. Install the Python dependencies
   pip install -r requirements.txt

   # 3. Install FlashAttention from your local wheel
   pip install /path/to/flash_attn-2.8.1+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl

.. note::

   Do not install ``flash-attn`` from PyPI — it would trigger a lengthy
   source build and may produce a binary incompatible with this PyTorch ABI.

External Resources: Pretrained Models
--------------------------------------

AlphaGenome (PyTorch)
~~~~~~~~~~~~~~~~~~~~~

Runs that use ``predict_model: alphagenome`` require the pretrained AlphaGenome
weights in PyTorch format. Download the
`gtca/alphagenome_pytorch <https://huggingface.co/gtca/alphagenome_pytorch>`_
repository from Hugging Face (~880 MB) to a local directory:

.. code-block:: bash

   # with the Hugging Face CLI
   huggingface-cli download gtca/alphagenome_pytorch \
       --local-dir /path/to/huggingface/gtca/alphagenome_pytorch

   # or, with newer versions of the CLI
   hf download gtca/alphagenome_pytorch \
       --local-dir /path/to/huggingface/gtca/alphagenome_pytorch

The downloaded directory should contain:

.. code-block:: text

   alphagenome_pytorch/
   ├── config.json                    # model architecture config
   ├── model_all_folds.safetensors    # pretrained weights (~880 MB)
   ├── LICENSE
   └── README.md

Then point ``predict_model_config`` in your run config (e.g.
``Runs/configs/adalead.yaml``) to the downloaded weights file:

.. code-block:: yaml

   predict_model: alphagenome
   predict_model_config:
     - /path/to/huggingface/gtca/alphagenome_pytorch/model_all_folds.safetensors

HyenaDNA (LongSafari weights + hyena-dna source code)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Runs that load the pretrained HyenaDNA foundation model (Ctrl-DNA, TACO, and
regLM) require **two** external resources:

1. The pretrained weights
   `LongSafari/hyenadna-medium-160k-seqlen <https://huggingface.co/LongSafari/hyenadna-medium-160k-seqlen>`_
   from Hugging Face:

   .. code-block:: bash

      huggingface-cli download LongSafari/hyenadna-medium-160k-seqlen \
          --local-dir /path/to/huggingface/LongSafari/hyenadna-medium-160k-seqlen

   The downloaded directory should contain:

   .. code-block:: text

      hyenadna-medium-160k-seqlen/
      ├── config.json      # model architecture config
      └── weights.ckpt     # pretrained Lightning checkpoint

2. A local clone of the `hyena-dna <https://github.com/HazyResearch/hyena-dna>`_
   GitHub repository — the model class (``ConvLMHeadModel``) is imported
   directly from its source tree:

   .. code-block:: bash

      git clone https://github.com/HazyResearch/hyena-dna /path/to/github/hyena-dna

Then update the hard-coded local paths where the regLM lightning modules
define the HyenaDNA locations:

- ``Tools/Ctrl-DNA/ctrl_dna/src/reglm/lightning.py`` — ``ckpt_dir`` (path to
  ``weights.ckpt``), ``hyenadna_path`` (path to the cloned repo), and
  ``config_path`` (path to ``config.json``)
- ``Tools/TACO/reglm_src/reglm/lightning.py`` — ``ckpt_dir`` (path to the
  downloaded weights directory) and ``hyenadna_path`` (path to the cloned repo)

DNA-Diffusion
--------------

`DNA-Diffusion <https://github.com/pinellolab/DNA-Diffusion>`_ uses
`uv <https://github.com/astral-sh/uv>`_ as its preferred package and project
manager. Install ``uv`` and download the upstream project configuration
(including ``pyproject.toml`` and ``uv.lock``) by following the official
installation guide:

https://github.com/pinellolab/DNA-Diffusion#installation

``Tools/DNA-Diffusion/`` in this repository stores only the model code modified
to bring DNA-Diffusion into the unified benchmark framework. It does not
replace the upstream checkout or its ``uv`` environment. The benchmark entry
point ``benchmark_DNADiffusion.py`` launches training with ``uv run``, so that
environment must already be in place.

Usage
-----

Step 1: Edit the run config
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Each benchmark method has a YAML config in ``Runs/configs/`` (e.g.
``Runs/configs/adalead.yaml``). Edit the config before launching:

- ``env``, ``script``, ``work_dir`` — the conda environment, the benchmark script,
  and its working directory (relative paths are resolved against the config
  file's directory)
- ``params`` — hyperparameters shared by all runs of this method (e.g.
  ``device_id``, ``length``, ``batch_size``, plus method-specific options)
- ``params.data`` — one entry per dataset; each entry is merged with the shared
  ``params`` and triggers one run. Key per-entry fields:

  - ``name`` — dataset/run name
  - ``save_dir`` — output directory for this run (see `Outputs`_)
  - ``predict_model``, ``predict_model_config``, ``predict_model_ckpt``,
    ``target_index``, ... — predictor settings

Step 2: Launch
~~~~~~~~~~~~~~

Run the launcher from ``Runs/launchers/``, passing one or more config names —
the file name under ``Runs/configs/`` without the ``.yaml`` extension:

.. code-block:: bash

   cd Runs/launchers
   python launcher.py adalead

This loads ``Runs/configs/adalead.yaml`` and executes one run per entry in its
``data`` list. You can also run several methods in one go:

.. code-block:: bash

   python launcher.py adalead fastseqprop simulatedannealing

Outputs
~~~~~~~

Everything produced by a run is stored in the ``save_dir`` of the corresponding
``data`` entry:

- ``model.pt``, ``generate_seqs.csv`` — results written by the benchmark script
- ``run.log`` — full stdout/stderr of the run
- ``config.yaml``, ``params.yaml`` — the exact config and merged parameters used
- ``time.txt`` — start/end time, duration, and peak RAM/GPU memory usage

.. note::

   Re-running the same config overwrites the files in ``save_dir``.
   Change ``save_dir`` (or back up the directory) to keep previous results.

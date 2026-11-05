# Fixed software environment

## Scope and provenance

The reference target is CPython **3.9.25 on Linux x86-64**. The manuscript
reports PyTorch **2.8.0 compiled for CUDA 12.8**, NumPy **2.0.2**, pandas
**2.3.1**, scikit-learn **1.6.1**, and Matplotlib **3.9.4**. These versions
are preserved rather than replaced with the newest packages.

`requirements/paper.txt` fixes the remaining direct, baseline, test, and
supporting Python dependencies. These are **release pins**, not a recovered
historical lockfile for every experiment. The supporting versions match the
available paper-core-version validation runtime where installed; pytest,
pluggy, iniconfig, PyYAML, and the build frontend are additional release test
and packaging dependencies. In particular,
the baseline pins are release choices, not evidence that every historical
baseline run used exactly those versions. Their declared Python and dependency
compatibility is checked against official package metadata.

This is a Python version-pinned environment, not a bit-for-bit operating-system
image or a wheel-hash lock. Conda build revisions, the NVIDIA driver, hardware,
and GPU reduction order are not frozen by the requirements files. Record these
along with the source commit for each new experiment.

## CUDA 12.8 reference installation

Run from the repository root in a fresh environment:

```bash
conda env create -f environment-paper.yml
conda activate fcm-ml-paper
python -m pip check
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The equivalent explicit steps are:

```bash
conda create -n fcm-ml-paper python=3.9.25 pip=25.2 -c conda-forge
conda activate fcm-ml-paper
python -m pip install -r requirements/paper-cu128.txt
python -m pip install --no-build-isolation --no-deps -e ".[baselines,dev]"
python -m pip check
```

`requirements/paper-cu128.txt` selects the CUDA **12.8** build explicitly;
an unqualified install of `torch` is not the reference installation.
PyTorch's CUDA libraries and Triton are version-fixed by the selected
PyTorch wheel's dependency metadata. A compatible NVIDIA driver is required
separately; installing the wheel does not install a host driver.
`torch.version.cuda` must be `12.8`, and `torch.cuda.is_available()` must be
`True` before starting GPU experiments. The pretraining launcher additionally
requires at least two visible GPUs and NCCL; the manuscript used eight GPUs.

`requirements.txt` is the full CUDA reference installation shorthand,
including the editable project. It is not a CPU-only installation file.
Do not install the CPU and CUDA requirement wrappers into the same environment.

## CPU regression and synthetic examples

For readers without GPUs, use the same scientific package versions with
the CPU PyTorch build:

```bash
conda create -n fcm-ml-cpu python=3.9.25 pip=25.2 -c conda-forge
conda activate fcm-ml-cpu
python -m pip install -r requirements/cpu.txt
python -m pip install --no-build-isolation --no-deps -e ".[baselines,dev]"
python -m pip check
OMP_NUM_THREADS=1 python -m pytest -q
python examples/synthetic_demo.py
```

The standard XGBoost 2.1.4 wheel depends on NCCL on Linux x86-64, so its
dependency may still be downloaded in this CPU profile. This does not select
a CUDA PyTorch build or require a GPU for CPU XGBoost inference.

The regression suite uses synthetic data and saved toy checkpoints; no
restricted clinical files are required. It checks code behavior, not the
published clinical scores. MAE distributed pretraining cannot run in the CPU
profile, and CPU execution is not a substitute for the paper's GPU timings.

The release validation also uses an available Python 3.9.25 runtime whose
six manuscript-reported core versions match exactly, with a CUDA-12.8-built
PyTorch 2.8.0 operating on CPU. This is not evidence of rerunning eight-GPU
pretraining, a fresh full CUDA installation, or recovering the original
experiment environment. See [validation.md](validation.md) for test scope.

## Record an experiment environment

```bash
git rev-parse HEAD
python --version
python -m pip freeze > outputs/environment-pip-freeze.txt
python -m pip check
nvidia-smi > outputs/environment-nvidia-smi.txt
```

Create `outputs/` first. Do not publish raw clinical files, private manifest
paths, or patient identifiers with these records. Preserve the package freeze
and hardware record privately with each clinical run when access is restricted.

## Official compatibility references

- [PyTorch 2.8.0 installation variants](https://pytorch.org/get-started/previous-versions/#v280)
- [PyTorch 2.8.0 package metadata](https://pypi.org/pypi/torch/2.8.0/json)
- [SciPy 1.13.1 package metadata](https://pypi.org/pypi/scipy/1.13.1/json)
- [Numba 0.60.0 package metadata](https://pypi.org/pypi/numba/0.60.0/json)
- [UMAP 0.5.12 package metadata](https://pypi.org/pypi/umap-learn/0.5.12/json)
- [XGBoost 2.1.4 package metadata](https://pypi.org/pypi/xgboost/2.1.4/json)

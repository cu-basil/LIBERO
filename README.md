# LIBERO + OpenPI Evaluation Suite

> **Benchmarking Knowledge Transfer for Lifelong Robot Learning**  
> Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone  
> [[Website]](https://libero-project.github.io) | [[Paper]](https://arxiv.org/pdf/2306.03310.pdf) | [[Docs]](https://lifelong-robot-learning.github.io/LIBERO/)

This fork integrates **Physical Intelligence's OpenPI** ($\pi_0$ / $\pi_{0.5}$) for native PyTorch evaluation on LIBERO tasks without websocket overhead.

---

## Quick Navigation

| Section | Description |
|---------|-------------|
| [Quick Start](#quick-start) | Get running in 5 minutes |
| [OpenPI Evaluation](#openpi-evaluation) | Evaluate $\pi_{0.5}$ on LIBERO |
| [Performance](#performance-optimizations) | 10x speedup breakdown |
| [Task Suites](#task-suites) | LIBERO benchmark overview |
| [Installation](#installation) | Full environment setup |
| [Training](#training) | Train your own policies |

---

## Quick Start

```bash
# 1. Clone with submodules
git clone --recurse-submodules <repo-url>
cd libero

# 2. Setup environment  
source scripts/use-repo-env.sh

# 3. Download checkpoints
mkdir -p .cache/pi05_libero
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_libero/* .cache/pi05_libero/

# 4. Convert weights (one-time)
source .venv_converter/bin/activate
python openpi/examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir .cache/pi05_libero \
    --config_name pi05_libero \
    --output_path .cache/pi05_libero

# 5. Run evaluation
./openpi_integration/evaluate_turbo.sh --all-suites
```

---

## OpenPI Evaluation

### Evaluation Scripts

| Script | Description | Speed |
|--------|-------------|-------|
| `evaluate_turbo.sh` | **Recommended** - 8 parallel envs, batched inference | ~10x faster |
| `evaluate_native.sh` | Single env, direct inference (no websocket) | ~2x faster |
| `evaluate_openpi_libero.sh` | Original websocket-based | Baseline |

### Usage Examples

```bash
# Run all 5 suites (libero_spatial, libero_object, libero_goal, libero_10, libero_90)
./openpi_integration/evaluate_turbo.sh --all-suites

# Single suite
./openpi_integration/evaluate_turbo.sh --task-suite-name libero_spatial

# Custom settings
./openpi_integration/evaluate_turbo.sh \
    --task-suite-name libero_spatial \
    --num-envs 4 \
    --num-trials-per-task 20 \
    --replan-steps 10 \
    --num-denoise-steps 5

# Background run with logging
nohup ./openpi_integration/evaluate_turbo.sh --all-suites > .cache/eval.log 2>&1 &
tail -f .cache/eval.log
```

### Python Direct Usage

```python
# Run directly without shell wrapper
MUJOCO_GL=egl python openpi_integration/libero_turbo_eval.py \
    --config-name pi05_libero \
    --checkpoint-dir .cache/pi05_libero \
    --task-suite-name libero_spatial \
    --num-envs 8 \
    --num-trials-per-task 50
```

### Output

Results saved to `docs/results/{suite}_results.json`:
```json
{
  "task_suite": "libero_spatial",
  "success_rates": {"task_0": 0.92, "task_1": 0.88, ...},
  "average_success_rate": 0.85,
  "total_time_seconds": 1847.3
}
```

---

## Performance Optimizations

**TURBO mode achieves ~10x speedup** over websocket baseline:

| Optimization | Speedup | Description |
|--------------|---------|-------------|
| Vectorized envs | 2x | 8 parallel SubprocVectorEnv instances |
| Full action horizon | 2x | Replan every 10 steps (model outputs 10) |
| Reduced denoising | 2x | 5 denoising steps instead of 10 |
| SDPA attention | 1.3x | PyTorch scaled_dot_product_attention |
| No websocket | 1.2x | Direct inference, no serialization |

**Benchmark (per inference):**
- Websocket baseline: ~750ms
- Native single-env: ~350ms  
- TURBO batched (8 envs): ~88ms/env effective

---

## Task Suites

| Suite | Tasks | Description |
|-------|-------|-------------|
| `libero_spatial` | 10 | Same objects, different spatial arrangements |
| `libero_object` | 10 | Same layout, different objects |
| `libero_goal` | 10 | Same scene, different goals |
| `libero_10` | 10 | Downstream lifelong learning test |
| `libero_90` | 90 | Pretraining tasks |

Each task has 50 fixed initial states for reproducible benchmarking.

---

## Installation

### Main Environment (.venv312)

```bash
# Create environment
"$(pyenv prefix 3.12.12)"/bin/python3.12 -m venv .venv312
source scripts/use-repo-env.sh

# Install dependencies
pip install --upgrade pip setuptools wheel
pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio
pip install numpy hydra-core wandb transformers opencv-python bddl einops
pip install h5py imageio imageio-ffmpeg matplotlib mujoco robosuite==1.4.0
pip install --no-deps robomimic==0.2.0
pip install -e .
```

### Converter Environment (.venv_converter)

Only needed for JAX→PyTorch weight conversion:

```bash
python3.12 -m venv .venv_converter
source .venv_converter/bin/activate
pip install jax[cuda12] flax orbax-checkpoint safetensors
```

### Environment Variables

`source scripts/use-repo-env.sh` sets:
- `LIBERO_ROOT` → repo root
- `HF_HOME`, `TORCH_HOME` → `.cache/`
- `LIBERO_DATASETS` → `./datasets`

---

## Datasets

```bash
# Download all suites
python benchmark_scripts/download_libero_datasets.py

# Or specific suite
python benchmark_scripts/download_libero_datasets.py --datasets libero_spatial

# From HuggingFace (faster)
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```

---

## Training

Train lifelong learning policies:

```bash
export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0

python -m libero.lifelong.main \
    seed=42 \
    benchmark_name=LIBERO_SPATIAL \
    policy=bc_transformer_policy \
    lifelong=er
```

Options:
- **Benchmarks**: `LIBERO_SPATIAL`, `LIBERO_OBJECT`, `LIBERO_GOAL`, `LIBERO_90`, `LIBERO_10`
- **Policies**: `bc_rnn_policy`, `bc_transformer_policy`, `bc_vilt_policy`
- **Algorithms**: `base`, `er`, `ewc`, `packnet`, `multitask`

---

## Project Structure

```
libero/
├── openpi_integration/      # OpenPI evaluation scripts
│   ├── evaluate_turbo.sh    # TURBO launcher (recommended)
│   ├── evaluate_native.sh   # Native single-env launcher
│   ├── libero_turbo_eval.py # Main turbo eval logic
│   └── libero_native_eval.py
├── libero/                  # Core LIBERO library
│   ├── lifelong/            # Lifelong learning algorithms
│   └── libero/              # Benchmark & environments
├── openpi/                  # OpenPI submodule
├── datasets/                # Demo datasets (HDF5)
├── .cache/                  # Checkpoints, logs, models
└── docs/results/            # Evaluation outputs
```

---

## Citation

```bibtex
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}
```

---

## License

| Component | License |
|-----------|---------|
| Codebase | [MIT](LICENSE) |
| Datasets | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| OpenPI | [Apache 2.0](openpi/LICENSE) |

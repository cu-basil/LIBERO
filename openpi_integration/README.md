# OpenPI Integration for LIBERO

Native PyTorch evaluation of $\pi_0$ / $\pi_{0.5}$ on LIBERO benchmarks without websocket overhead.

## Quick Start

```bash
# Run all 5 suites
./evaluate_turbo.sh --all-suites

# Single suite
./evaluate_turbo.sh --task-suite-name libero_spatial
```

## Evaluation Scripts

| Script | Description | Speed |
|--------|-------------|-------|
| `evaluate_turbo.sh` | **Recommended** - 8 parallel envs, batched inference | ~10x faster |
| `evaluate_native.sh` | Single env, direct inference | ~2x faster |
| `evaluate_openpi_libero.sh` | Original websocket-based | Baseline |

## Performance (TURBO mode)

| Optimization | Speedup |
|--------------|---------|
| Vectorized envs (8 parallel) | 2x |
| Full action horizon (replan=10) | 2x |
| Reduced denoising (5 steps) | 2x |
| SDPA attention | 1.3x |
| No websocket | 1.2x |
| **Total** | **~10x** |

## Usage

```bash
# Custom settings
./evaluate_turbo.sh \
    --task-suite-name libero_spatial \
    --num-envs 4 \
    --num-trials-per-task 20 \
    --replan-steps 10 \
    --num-denoise-steps 5

# Background run
nohup ./evaluate_turbo.sh --all-suites > ../.cache/eval.log 2>&1 &
tail -f ../.cache/eval.log
```

## Setup

1. Download and convert checkpoints:
```bash
mkdir -p .cache/pi05_libero
gsutil -m cp -r gs://openpi-assets/checkpoints/pi05_libero/* .cache/pi05_libero/

source .venv_converter/bin/activate
python openpi/examples/convert_jax_model_to_pytorch.py \
    --checkpoint_dir .cache/pi05_libero \
    --config_name pi05_libero \
    --output_path .cache/pi05_libero
```

2. Run evaluation from repo root:
```bash
./openpi_integration/evaluate_turbo.sh --all-suites
```

## Output

Results saved to `docs/results/{suite}_results.json`:
```json
{
  "task_suite": "libero_spatial",
  "success_rates": {"task_0": 0.92, ...},
  "average_success_rate": 0.85
}
```

## Files

- `libero_turbo_eval.py` - Main turbo evaluation (parallel envs + batched inference)
- `libero_native_eval.py` - Single-env native inference
- `libero_client_eval.py` - Original websocket client
- `batched_policy.py` - Batched policy wrapper for vectorized envs
- `evaluate_*.sh` - Shell launchers

## Task Suites

| Suite | Tasks |
|-------|-------|
| `libero_spatial` | 10 |
| `libero_object` | 10 |
| `libero_goal` | 10 |
| `libero_10` | 10 |
| `libero_90` | 90 |

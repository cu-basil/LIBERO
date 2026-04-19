"""
Benchmark script to measure speedup from various optimizations.

Tests:
1. Baseline (websocket, 5 replan, 10 denoise)
2. Native (no websocket)
3. Native + 10 replan
4. Native + 5 denoise
5. Native + 10 replan + 5 denoise
6. Batched inference

Usage:
    python openpi_integration/benchmark_speedup.py
"""

import dataclasses
import logging
import time
from typing import Optional

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, force=True)


@dataclasses.dataclass
class BenchmarkResult:
    name: str
    total_time_ms: float
    num_inferences: int
    avg_infer_ms: float
    throughput: float  # inferences per second
    
    def __str__(self):
        return (
            f"{self.name:40s}: "
            f"avg={self.avg_infer_ms:6.2f}ms, "
            f"throughput={self.throughput:6.1f}/s, "
            f"total={self.total_time_ms/1000:.2f}s"
        )


def benchmark_single_inference(policy, num_runs: int = 50) -> BenchmarkResult:
    """Benchmark single observation inference."""
    
    # Create dummy observation
    obs = {
        "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
        "observation/state": np.random.randn(8).astype(np.float32),
        "prompt": "pick up the black bowl and place it on the plate",
    }
    
    # Warmup
    logging.info("Warming up (3 inferences)...")
    for _ in range(3):
        policy.infer(obs)
    
    # Benchmark
    logging.info(f"Running {num_runs} inferences...")
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        policy.infer(obs)
        times.append((time.perf_counter() - start) * 1000)
    
    total_ms = sum(times)
    avg_ms = np.mean(times)
    throughput = 1000 / avg_ms
    
    return BenchmarkResult(
        name="Single inference",
        total_time_ms=total_ms,
        num_inferences=num_runs,
        avg_infer_ms=avg_ms,
        throughput=throughput,
    )


def benchmark_batched_inference(batched_policy, batch_size: int, num_runs: int = 20) -> BenchmarkResult:
    """Benchmark batched inference."""
    
    # Create batch of dummy observations
    obs_batch = []
    for _ in range(batch_size):
        obs = {
            "observation/image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8),
            "observation/state": np.random.randn(8).astype(np.float32),
            "prompt": "pick up the black bowl and place it on the plate",
        }
        obs_batch.append(obs)
    
    # Warmup
    logging.info(f"Warming up batched (batch_size={batch_size})...")
    for _ in range(2):
        batched_policy.infer_batch(obs_batch)
    
    # Benchmark
    logging.info(f"Running {num_runs} batched inferences...")
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        batched_policy.infer_batch(obs_batch)
        times.append((time.perf_counter() - start) * 1000)
    
    total_ms = sum(times)
    total_inferences = num_runs * batch_size
    avg_ms_per_batch = np.mean(times)
    avg_ms_per_sample = avg_ms_per_batch / batch_size
    throughput = batch_size * 1000 / avg_ms_per_batch
    
    return BenchmarkResult(
        name=f"Batched (batch_size={batch_size})",
        total_time_ms=total_ms,
        num_inferences=total_inferences,
        avg_infer_ms=avg_ms_per_sample,
        throughput=throughput,
    )


def run_benchmarks():
    """Run all benchmarks and report results."""
    
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    from batched_policy import BatchedPolicyWrapper
    
    results = []
    
    # ========================================================================
    # Load models with different configs
    # ========================================================================
    checkpoint_dir = ".cache/pi05_libero"
    config_name = "pi05_libero"
    
    logging.info("="*70)
    logging.info("LOADING MODELS...")
    logging.info("="*70)
    
    # Baseline: 10 denoise steps
    logging.info("Loading baseline model (10 denoise steps)...")
    train_config = _config.get_config(config_name)
    policy_10 = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        pytorch_device="cuda",
        sample_kwargs={"num_steps": 10},
    )
    
    # Optimized: 5 denoise steps
    logging.info("Loading optimized model (5 denoise steps)...")
    policy_5 = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        pytorch_device="cuda",
        sample_kwargs={"num_steps": 5},
    )
    
    # Even faster: 3 denoise steps
    logging.info("Loading fast model (3 denoise steps)...")
    policy_3 = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        pytorch_device="cuda",
        sample_kwargs={"num_steps": 3},
    )
    
    # Batched policy wrapper
    batched_policy = BatchedPolicyWrapper(policy_5)
    
    logging.info("="*70)
    logging.info("RUNNING BENCHMARKS...")
    logging.info("="*70)
    
    # ========================================================================
    # Run benchmarks
    # ========================================================================
    
    # Single inference with different denoise steps
    results.append(benchmark_single_inference(policy_10, num_runs=30))
    results[-1].name = "Single (10 denoise)"
    
    results.append(benchmark_single_inference(policy_5, num_runs=50))
    results[-1].name = "Single (5 denoise)"
    
    results.append(benchmark_single_inference(policy_3, num_runs=50))
    results[-1].name = "Single (3 denoise)"
    
    # Batched inference
    for batch_size in [2, 4, 8]:
        results.append(benchmark_batched_inference(batched_policy, batch_size, num_runs=20))
    
    # ========================================================================
    # Report results
    # ========================================================================
    
    logging.info("\n" + "="*70)
    logging.info("BENCHMARK RESULTS")
    logging.info("="*70)
    
    baseline = results[0]
    for r in results:
        speedup = baseline.avg_infer_ms / r.avg_infer_ms
        print(f"{r} (speedup: {speedup:.2f}x)")
    
    logging.info("="*70)
    logging.info("SPEEDUP ANALYSIS")
    logging.info("="*70)
    
    # Calculate total speedup for evaluation scenario
    # Baseline: 10 denoise, 5 replan → 44 inferences per episode
    # Optimized: 5 denoise, 10 replan, batch=8 → 22 inferences per 8 episodes
    
    baseline_per_ep = baseline.avg_infer_ms * (220 / 5)  # ~44 inferences
    
    # With 10 replan, 5 denoise
    opt_single = results[1].avg_infer_ms * (220 / 10)  # ~22 inferences
    
    # With batching (8 envs)
    batch_8 = next(r for r in results if "batch_size=8" in r.name)
    opt_batched = batch_8.avg_infer_ms * (220 / 10)  # per env, ~22 inferences
    
    print(f"\nPer-episode inference time:")
    print(f"  Baseline (10 denoise, 5 replan):     {baseline_per_ep:.0f}ms")
    print(f"  Optimized (5 denoise, 10 replan):    {opt_single:.0f}ms ({baseline_per_ep/opt_single:.1f}x)")
    print(f"  Batched (5 denoise, 10 replan, b=8): {opt_batched:.0f}ms ({baseline_per_ep/opt_batched:.1f}x)")
    
    logging.info("="*70)


if __name__ == "__main__":
    run_benchmarks()

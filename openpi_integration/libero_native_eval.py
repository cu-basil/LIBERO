"""
Direct LIBERO evaluation without websocket server overhead.

This bypasses the client-server architecture entirely by loading the OpenPI policy
directly in the same process as the LIBERO environment. This eliminates:
- Websocket serialization/deserialization latency
- msgpack encoding/decoding for images (~300KB per observation)
- IPC overhead between separate processes
- Asyncio event loop context switching

Usage:
    python openpi_integration/libero_native_eval.py --task-suite-name libero_spatial
"""

import collections
import dataclasses
import logging
import pathlib
import time
from typing import Optional

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
import tqdm
import tyro

# Direct imports from openpi - no websocket needed
from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256


@dataclasses.dataclass
class Args:
    """Arguments for native LIBERO evaluation."""
    
    # Model configuration
    config_name: str = "pi05_libero"
    checkpoint_dir: str = ".cache/pi05_libero"
    resize_size: int = 224
    replan_steps: int = 5
    pytorch_device: str = "cuda"
    
    # LIBERO environment parameters
    task_suite_name: str = "libero_spatial"
    num_steps_wait: int = 10
    num_trials_per_task: int = 50
    
    # Output
    video_out_path: str = "data/libero/videos"
    results_path: str = "docs/results"
    seed: int = 7
    
    # Timing instrumentation
    collect_timing: bool = True


class TimingStats:
    """Collect and report timing statistics."""
    
    def __init__(self):
        self.infer_times = []
        self.env_step_times = []
        self.preprocess_times = []
        self.total_episode_times = []
    
    def add_infer(self, t: float):
        self.infer_times.append(t * 1000)  # ms
    
    def add_env_step(self, t: float):
        self.env_step_times.append(t * 1000)
    
    def add_preprocess(self, t: float):
        self.preprocess_times.append(t * 1000)
    
    def add_episode(self, t: float):
        self.total_episode_times.append(t)
    
    def report(self) -> str:
        def stats(arr):
            if not arr:
                return "N/A"
            arr = np.array(arr)
            return f"mean={arr.mean():.2f}ms, std={arr.std():.2f}ms, min={arr.min():.2f}ms, max={arr.max():.2f}ms"
        
        return (
            f"\n{'='*60}\n"
            f"TIMING STATISTICS (Native Inference - No Websocket)\n"
            f"{'='*60}\n"
            f"Model inference: {stats(self.infer_times)}\n"
            f"Image preprocessing: {stats(self.preprocess_times)}\n"
            f"Environment step: {stats(self.env_step_times)}\n"
            f"Total episodes: {len(self.total_episode_times)}\n"
            f"Avg episode time: {np.mean(self.total_episode_times):.2f}s\n"
            f"Total inferences: {len(self.infer_times)}\n"
            f"{'='*60}\n"
        )


def _quat2axisangle(quat):
    """Convert quaternion to axis-angle representation."""
    # Normalize quaternion
    quat = np.array(quat)
    quat = quat / np.linalg.norm(quat)
    
    # Extract angle
    angle = 2 * np.arccos(np.clip(quat[3], -1, 1))
    
    # Handle small angles
    if angle < 1e-6:
        return np.zeros(3)
    
    # Extract axis
    axis = quat[:3] / np.sin(angle / 2)
    return axis * angle


def _get_libero_env(task, resolution, seed):
    """Initialize LIBERO environment."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def get_max_steps(task_suite_name: str) -> int:
    """Get max steps for each task suite."""
    max_steps_map = {
        "libero_spatial": 220,
        "libero_object": 280,
        "libero_goal": 300,
        "libero_10": 520,
        "libero_90": 400,
    }
    if task_suite_name not in max_steps_map:
        raise ValueError(f"Unknown task suite: {task_suite_name}")
    return max_steps_map[task_suite_name]


def eval_libero_native(args: Args) -> None:
    """Run LIBERO evaluation with direct model inference (no websocket)."""
    
    np.random.seed(args.seed)
    timing = TimingStats() if args.collect_timing else None
    
    # ========================================================================
    # DIRECT MODEL LOADING - No server startup needed!
    # ========================================================================
    logging.info(f"Loading model directly from {args.checkpoint_dir}...")
    load_start = time.monotonic()
    
    train_config = _config.get_config(args.config_name)
    policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        pytorch_device=args.pytorch_device,
    )
    
    load_time = time.monotonic() - load_start
    logging.info(f"Model loaded in {load_time:.2f}s (direct, no server overhead)")
    
    # ========================================================================
    # Initialize LIBERO benchmark
    # ========================================================================
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    max_steps = get_max_steps(args.task_suite_name)
    
    logging.info(f"Task suite: {args.task_suite_name} ({num_tasks_in_suite} tasks)")
    
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.results_path).mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # Evaluation loop
    # ========================================================================
    total_episodes, total_successes = 0, 0
    
    for task_id in tqdm.tqdm(range(num_tasks_in_suite), desc="Tasks"):
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)
        
        task_episodes, task_successes = 0, 0
        
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task), desc="Episodes", leave=False):
            episode_start = time.monotonic()
            logging.info(f"\nTask: {task_description}")
            
            env.reset()
            action_plan = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])
            
            t = 0
            replay_images = []
            done = False
            
            logging.info(f"Starting episode {task_episodes+1}...")
            
            while t < max_steps + args.num_steps_wait:
                try:
                    # Wait for objects to stabilize
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue
                    
                    # ========================================================
                    # Preprocess observation
                    # ========================================================
                    preprocess_start = time.monotonic()
                    
                    # Rotate 180 degrees to match training preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    
                    if timing:
                        timing.add_preprocess(time.monotonic() - preprocess_start)
                    
                    replay_images.append(img)
                    
                    # ========================================================
                    # Get action from model - DIRECT CALL, no websocket!
                    # ========================================================
                    if not action_plan:
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate((
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )),
                            "prompt": str(task_description),
                        }
                        
                        infer_start = time.monotonic()
                        # DIRECT inference - no serialization, no websocket!
                        result = policy.infer(element)
                        if timing:
                            timing.add_infer(time.monotonic() - infer_start)
                        
                        action_chunk = result["actions"]
                        assert len(action_chunk) >= args.replan_steps
                        action_plan.extend(action_chunk[:args.replan_steps])
                    
                    action = action_plan.popleft()
                    
                    # ========================================================
                    # Step environment
                    # ========================================================
                    step_start = time.monotonic()
                    obs, reward, done, info = env.step(action.tolist())
                    if timing:
                        timing.add_env_step(time.monotonic() - step_start)
                    
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    
                    t += 1
                    
                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            
            task_episodes += 1
            total_episodes += 1
            
            if timing:
                timing.add_episode(time.monotonic() - episode_start)
            
            # Save replay video
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )
            
            logging.info(f"Success: {done}")
            logging.info(f"Episodes: {total_episodes}, Successes: {total_successes} ({total_successes/total_episodes*100:.1f}%)")
        
        logging.info(f"Task success rate: {task_successes/task_episodes:.2%}")
        logging.info(f"Total success rate: {total_successes/total_episodes:.2%}")
    
    # ========================================================================
    # Final results
    # ========================================================================
    final_rate = total_successes / total_episodes
    logging.info(f"\n{'='*60}")
    logging.info(f"FINAL RESULTS - {args.task_suite_name}")
    logging.info(f"{'='*60}")
    logging.info(f"Total success rate: {final_rate:.2%}")
    logging.info(f"Total episodes: {total_episodes}")
    
    if timing:
        logging.info(timing.report())
    
    # Save results
    results_file = pathlib.Path(args.results_path) / f"{args.task_suite_name}_native.log"
    with open(results_file, "w") as f:
        f.write(f"Task suite: {args.task_suite_name}\n")
        f.write(f"Success rate: {final_rate:.4f}\n")
        f.write(f"Total episodes: {total_episodes}\n")
        f.write(f"Total successes: {total_successes}\n")
        if timing:
            f.write(timing.report())
    
    logging.info(f"Results saved to {results_file}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    args = tyro.cli(Args)
    eval_libero_native(args)

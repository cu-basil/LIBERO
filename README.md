<div align="center">
<img src="https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/images/libero_logo.png" width="360">


<p align="center">
<a href="https://github.com/Lifelong-Robot-Learning/LIBERO/actions">
<img alt="Tests Passing" src="https://github.com/anuraghazra/github-readme-stats/workflows/Test/badge.svg" />
</a>
<a href="https://github.com/Lifelong-Robot-Learning/LIBERO/graphs/contributors">
<img alt="GitHub Contributors" src="https://img.shields.io/github/contributors/Lifelong-Robot-Learning/LIBERO" />
</a>
<a href="https://github.com/Lifelong-Robot-Learning/LIBERO/issues">
<img alt="Issues" src="https://img.shields.io/github/issues/Lifelong-Robot-Learning/LIBERO?color=0088ff" />

## **Benchmarking Knowledge Transfer for Lifelong Robot Learning**

Bo Liu, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, Peter Stone

[[Website]](https://libero-project.github.io)
[[Paper]](https://arxiv.org/pdf/2306.03310.pdf)
[[Docs]](https://lifelong-robot-learning.github.io/LIBERO/)
______________________________________________________________________
![pull_figure](https://github.com/Lifelong-Robot-Learning/LIBERO/blob/master/images//fig1.png)
</div>

**LIBERO** is designed for studying knowledge transfer in multitask and lifelong robot learning problems. Successfully resolving these problems require both declarative knowledge about objects/spatial relationships and procedural knowledge about motion/behaviors. **LIBERO** provides:
- a procedural generation pipeline that could in principle generate an infinite number of manipulation tasks.
- 130 tasks grouped into four task suites: **LIBERO-Spatial**, **LIBERO-Object**, **LIBERO-Goal**, and **LIBERO-100**. The first three task suites have controlled distribution shifts, meaning that they require the transfer of a specific type of knowledge. In contrast, **LIBERO-100** consists of 100 manipulation tasks that require the transfer of entangled knowledge. **LIBERO-100** is further splitted into **LIBERO-90** for pretraining a policy and **LIBERO-10** for testing the agent's downstream lifelong learning performance.
- five research topics.
- three visuomotor policy network architectures.
- three lifelong learning algorithms with the sequential finetuning and multitask learning baselines.

---


# Contents

- [Installation](#Installation)
- [Datasets](#Dataset)
- [Getting Started](#Getting-Started)
  - [Task](#Task)
  - [Training](#Training)
  - [Evaluation](#Evaluation)
- [Citation](#Citation)
- [License](#License)


# Installtion
The repo is now standardized on a repo-local Python 3.12 environment for both dataset work and simulator-backed fine-tuning. This keeps system Python and system packages untouched while storing pip, Hugging Face, torch, and temporary build files on the same drive as the repository.

```shell
"$(pyenv prefix 3.12.12)"/bin/python3.12 -m venv .venv312
source scripts/use-repo-env.sh
python -m pip install --upgrade pip setuptools wheel
python -m pip install --index-url https://download.pytorch.org/whl/cu130 torch torchvision torchaudio
python -m pip install numpy hydra-core wandb easydict transformers opencv-python bddl einops thop cloudpickle gym future
python -m pip install h5py psutil tensorboard tensorboardX imageio imageio-ffmpeg matplotlib termcolor
python -m pip install mujoco robosuite==1.4.0
python -m pip install --no-deps robomimic==0.2.0
python -m pip install -e .
```

`source scripts/use-repo-env.sh` stores the active caches under `.cache/`, keeps LIBERO's config under `.libero/`, and points the default dataset location at `./datasets`.

If your repository lives on a filesystem mounted with `noexec`, keep the caches in the repo but create the virtual environment on an exec-capable filesystem instead:

```shell
"$(pyenv prefix 3.12.12)"/bin/python3.12 -m venv "$HOME/.venvs/libero"
LIBERO_VENV_PATH_OVERRIDE="$HOME/.venvs/libero" source scripts/use-repo-env.sh
python -m pip install --upgrade pip setuptools wheel
```

# Smoke Tests

One optimizer step on a downloaded demo file:

```shell
source scripts/use-repo-env.sh
python scripts/smoke_train_step.py --benchmark LIBERO_SPATIAL --task-id 0 --steps 1
```

Minimal simulator reset:

```shell
source scripts/use-repo-env.sh
python scripts/smoke_env_reset.py --benchmark libero_spatial --task-id 0
```

# Datasets
We provide high-quality human teleoperation demonstrations for the four task suites in **LIBERO**. To download the demonstration dataset, run:
```python
python benchmark_scripts/download_libero_datasets.py
```
By default, the dataset will be stored under the ```LIBERO``` folder and all four datasets will be downloaded. To download a specific dataset, use
```python
python benchmark_scripts/download_libero_datasets.py --datasets DATASET
```
where ```DATASET``` is chosen from `[libero_spatial, libero_object, libero_100, libero_goal`.

**NEW!!!**

Alternatively, you can download the dataset from HuggingFace by using:
```python
python benchmark_scripts/download_libero_datasets.py --use-huggingface
```

This option can also be combined with the specific dataset selection:
```python
python benchmark_scripts/download_libero_datasets.py --datasets DATASET --use-huggingface
```

The datasets hosted on HuggingFace are available at [here](https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets).


# Getting Started

For a detailed walk-through, please either refer to the documentation or the notebook examples provided under the `notebooks` folder. In the following, we provide example scripts for retrieving a task, training and evaluation.

## Task

The following is a minimal example of retrieving a specific task from a specific task suite.
```python
from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv


benchmark_dict = benchmark.get_benchmark_dict()
task_suite_name = "libero_10" # can also choose libero_spatial, libero_object, etc.
task_suite = benchmark_dict[task_suite_name]()

# retrieve a specific task
task_id = 0
task = task_suite.get_task(task_id)
task_name = task.name
task_description = task.language
task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
print(f"[info] retrieving task {task_id} from suite {task_suite_name}, the " + \
      f"language instruction is {task_description}, and the bddl file is {task_bddl_file}")

# step over the environment
env_args = {
    "bddl_file_name": task_bddl_file,
    "camera_heights": 128,
    "camera_widths": 128
}
env = OffScreenRenderEnv(**env_args)
env.seed(0)
env.reset()
init_states = task_suite.get_task_init_states(task_id) # for benchmarking purpose, we fix the a set of initial states
init_state_id = 0
env.set_init_state(init_states[init_state_id])

dummy_action = [0.] * 7
for step in range(10):
    obs, reward, done, info = env.step(dummy_action)
env.close()
```
Currently, we only support sparse reward function (i.e., the agent receives `+1` when the task is finished). As sparse-reward RL is extremely hard to learn, currently we mainly focus on lifelong imitation learning.

## Training
To start a lifelong learning experiment, please choose:
- `BENCHMARK` from `[LIBERO_SPATIAL, LIBERO_OBJECT, LIBERO_GOAL, LIBERO_90, LIBERO_10]`
- `POLICY` from `[bc_rnn_policy, bc_transformer_policy, bc_vilt_policy]`
- `ALGO` from `[base, er, ewc, packnet, multitask]`

then run the following:

```shell
export CUDA_VISIBLE_DEVICES=GPU_ID && \
export MUJOCO_EGL_DEVICE_ID=GPU_ID && \
python -m libero.lifelong.main seed=SEED \
                               benchmark_name=BENCHMARK \
                               policy=POLICY \
                               lifelong=ALGO
```
Please see the documentation for the details of reproducing the study results.

## Evaluation

By default the policies will be evaluated on the fly during training. If you have limited computing resource of GPUs, we offer an evaluation script for you to evaluate models separately.

```shell
python -m libero.lifelong.evaluate --benchmark BENCHMARK_NAME \
                                    --task_id TASK_ID \ 
                                    --algo ALGO_NAME \
                                    --policy POLICY_NAME \
                                    --seed SEED \
                                    --ep EPOCH \
                                    --load_task LOAD_TASK \
                                    --device_id CUDA_ID
```

# Citation
If you find **LIBERO** to be useful in your own research, please consider citing our paper:

```bibtex
@article{liu2023libero,
  title={LIBERO: Benchmarking Knowledge Transfer for Lifelong Robot Learning},
  author={Liu, Bo and Zhu, Yifeng and Gao, Chongkai and Feng, Yihao and Liu, Qiang and Zhu, Yuke and Stone, Peter},
  journal={arXiv preprint arXiv:2306.03310},
  year={2023}
}
```

# License
| Component        | License                                                                                                                             |
|------------------|-------------------------------------------------------------------------------------------------------------------------------------|
| Codebase         | [MIT License](LICENSE)                                                                                                                      |
| Datasets         | [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/legalcode)                 |

---
## Integrating OpenPI ($\pi_0$ / $\pi_{0.5}$) Checkpoints

We have integrated Physical Intelligence's **OpenPI** repository as a submodule to natively evaluate the $\pi_{0.5}$ base model directly onto our LIBERO task suites without relying on their Docker containers or their default JAX interference stack. 

### Submodule Setup
Since `openpi` is integrated as a git submodule, if you clone this repository freshly, you must run:
```bash
git submodule update --init --recursive
```

### Virtual Environments
Due to dependency conflicts between `robosuite`/`mujoco` requirements and OpenPI's latest HuggingFace/JAX patches, the evaluation stack employs two distinct `uv` environments:
1. `.venv312` - Contains our main LIBERO stack, `robosuite`, and the simulation client.
2. `.venv_converter` - An isolated environment used only to host OpenPI's `serve_policy` PyTorch server and convert their JAX-native `.ocdbt` checkpoints to `model.safetensors`.

### Running Evaluation 
We've abstracted the OpenPI evaluation loop out of their repo into our native `openpi_integration` folder. To evaluate the 12GB checkpoint headlessly via EGL (and skip any "Address already in use" websocket locks), run the native bash script:

```bash
./openpi_integration/evaluate_openpi_libero.sh
```

**What this does:**
1. Spins up the $\pi_{0.5}$ websocket server locally binding to port 8011 inside `.venv_converter`.
2. Delays for 40 seconds to allow the heavy 12GB weights to initialize fully in PyTorch.
3. Automatically launches the LIBERO headless client directly in `.venv312` and pushes tasks through the server.
4. Results are simultaneously teed to `.cache/libero_client_eval.log` and the server process is aggressively cleaned up when done.

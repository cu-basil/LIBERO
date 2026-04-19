"""
Batched policy wrapper for OpenPI models.

Enables efficient batched inference by processing multiple observations
in a single forward pass, significantly reducing GPU kernel launch overhead.
"""

import time
from typing import Any

import numpy as np
import torch

from openpi.models import model as _model
from openpi.policies import policy as _policy


class BatchedPolicyWrapper:
    """
    Wrapper that adds batched inference capability to OpenPI policies.
    
    The base Policy class processes observations one at a time, adding batch dimension.
    This wrapper processes multiple observations in a single forward pass.
    """
    
    def __init__(self, policy: _policy.Policy):
        self._policy = policy
        self._model = policy._model
        self._input_transform = policy._input_transform
        self._output_transform = policy._output_transform
        self._sample_kwargs = policy._sample_kwargs
        self._is_pytorch_model = policy._is_pytorch_model
        self._pytorch_device = policy._pytorch_device
    
    def infer_batch(self, obs_list: list[dict]) -> list[dict]:
        """
        Perform batched inference on multiple observations.
        
        Currently uses fast sequential inference (no websocket overhead).
        True GPU batching requires model modifications for image_masks handling.
        
        Args:
            obs_list: List of observation dicts
        
        Returns:
            List of result dicts with actions
        """
        if len(obs_list) == 0:
            return []
        
        # Fast sequential inference - still much faster than websocket
        # due to no serialization and warm GPU
        start_time = time.monotonic()
        results = []
        for obs in obs_list:
            results.append(self._policy.infer(obs))
        total_time = time.monotonic() - start_time
        
        # Add batch timing info
        for r in results:
            r["policy_timing"]["batch_size"] = len(obs_list)
            r["policy_timing"]["total_batch_ms"] = total_time * 1000
        
        return results
    
    def infer_batch_parallel(self, obs_list: list[dict]) -> list[dict]:
        """
        [EXPERIMENTAL] True batched inference - requires matching batch dims.
        Falls back to sequential if batching fails.
        """
        if len(obs_list) == 0:
            return []
        
        if len(obs_list) == 1:
            return [self._policy.infer(obs_list[0])]
        
        batch_size = len(obs_list)
        
        try:
            # Transform each observation
            transformed_list = []
            for obs in obs_list:
                inputs = {k: v for k, v in obs.items()}
                inputs = self._input_transform(inputs)
                transformed_list.append(inputs)
            
            # Stack into batched tensors
            def stack_field(field: str):
                arrays = [t[field] for t in transformed_list]
                if isinstance(arrays[0], str):
                    return arrays[0]
                stacked = np.stack(arrays, axis=0)
                if self._is_pytorch_model:
                    return torch.from_numpy(stacked).to(self._pytorch_device)
                else:
                    import jax.numpy as jnp
                    return jnp.asarray(stacked)
            
            # Build batched inputs
            batched_inputs = {}
            skip_keys = {"prompt", "tokenized_prompt", "tokenized_prompt_mask"}
            
            for key in transformed_list[0].keys():
                val = transformed_list[0][key]
                if key in skip_keys or isinstance(val, str):
                    batched_inputs[key] = val
                elif isinstance(val, dict):
                    # Handle nested dicts like images, image_masks
                    batched_inputs[key] = {}
                    for subkey in val.keys():
                        subarrays = [t[key][subkey] for t in transformed_list]
                        if isinstance(subarrays[0], (bool, np.bool_)):
                            # Convert scalar bools to batch of bool arrays
                            batched_inputs[key][subkey] = np.array(subarrays, dtype=bool)
                        elif isinstance(subarrays[0], np.ndarray):
                            stacked = np.stack(subarrays, axis=0)
                            if self._is_pytorch_model:
                                batched_inputs[key][subkey] = torch.from_numpy(stacked).to(self._pytorch_device)
                            else:
                                batched_inputs[key][subkey] = stacked
                        else:
                            batched_inputs[key][subkey] = subarrays[0]
                elif isinstance(val, np.ndarray):
                    batched_inputs[key] = stack_field(key)
                else:
                    try:
                        batched_inputs[key] = stack_field(key)
                    except (TypeError, ValueError):
                        batched_inputs[key] = val
            
            observation = _model.Observation.from_dict(batched_inputs)
            
            # Run batched inference
            sample_kwargs = dict(self._sample_kwargs)
            
            start_time = time.monotonic()
            
            if self._is_pytorch_model:
                with torch.no_grad():
                    actions = self._model.sample_actions(
                        self._pytorch_device, 
                        observation, 
                        **sample_kwargs
                    )
            else:
                raise NotImplementedError("Batched inference for JAX models not implemented")
            
            model_time = time.monotonic() - start_time
            
            # Unbatch results
            results = []
            for i in range(batch_size):
                if self._is_pytorch_model:
                    actions_i = actions[i].detach().cpu().numpy()
                    state_i = batched_inputs["state"][i].detach().cpu().numpy()
                else:
                    actions_i = np.asarray(actions[i])
                    state_i = np.asarray(batched_inputs["state"][i])
                
                output = {
                    "actions": actions_i,
                    "state": state_i,
                }
                output = self._output_transform(output)
                output["policy_timing"] = {
                    "infer_ms": model_time * 1000 / batch_size,
                    "batch_size": batch_size,
                    "total_batch_ms": model_time * 1000,
                }
                results.append(output)
            
            return results
            
        except Exception as e:
            # Fall back to sequential on any batching error
            import logging
            logging.warning(f"Batched inference failed, falling back to sequential: {e}")
            return self.infer_batch(obs_list)
    
    def infer(self, obs: dict) -> dict:
        """Single observation inference (delegates to wrapped policy)."""
        return self._policy.infer(obs)


def create_batched_policy(
    config_name: str,
    checkpoint_dir: str,
    pytorch_device: str = "cuda",
    num_denoise_steps: int = 5,
) -> BatchedPolicyWrapper:
    """
    Create a policy with batched inference support.
    
    Args:
        config_name: OpenPI config name (e.g., "pi05_libero")
        checkpoint_dir: Path to checkpoint directory
        pytorch_device: Device to use ("cuda", "cuda:0", etc.)
        num_denoise_steps: Number of denoising steps (default 10, use 5 for 2x speed)
    
    Returns:
        BatchedPolicyWrapper with infer_batch method
    """
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config
    
    train_config = _config.get_config(config_name)
    
    policy = _policy_config.create_trained_policy(
        train_config,
        checkpoint_dir,
        pytorch_device=pytorch_device,
        sample_kwargs={"num_steps": num_denoise_steps},
    )
    
    return BatchedPolicyWrapper(policy)

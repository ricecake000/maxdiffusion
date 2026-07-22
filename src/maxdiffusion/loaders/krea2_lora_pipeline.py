# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Inference-time LoRA loading for the Krea 2 Flax transformer. Follows the Flux
# Linen interceptor pattern: LoRA tensors live under a `lora-{adapter}` subtree
# inside each target Dense's params, and an `nn.intercept_methods` interceptor
# adds the low-rank update to every intercepted `Dense.__call__`.

import os
from typing import Dict, Union

import jax.numpy as jnp
from flax.traverse_util import flatten_dict, unflatten_dict

from .. import max_logging
from ..models.krea2.lora_util import convert_krea2_lora_to_flax
from ..models.lora import BaseLoRALayer, LoRALinearLayer
from .lora_base import LoRABaseMixin


class Krea2LoraLoaderMixin(LoRABaseMixin):
  _lora_lodable_modules = ["transformer"]

  @classmethod
  def lora_state_dict(cls, pretrained_model_name_or_path_or_dict: Union[str, Dict], weight_name=None, **kwargs):
    if isinstance(pretrained_model_name_or_path_or_dict, dict):
      return pretrained_model_name_or_path_or_dict

    import safetensors.torch

    path = pretrained_model_name_or_path_or_dict
    if os.path.isfile(path):
      return safetensors.torch.load_file(path, device="cpu")
    if os.path.isdir(path) and weight_name and os.path.isfile(os.path.join(path, weight_name)):
      return safetensors.torch.load_file(os.path.join(path, weight_name), device="cpu")

    # Fall back to the shared HF Hub fetch (repo id + weight_name).
    return cls._fetch_state_dict(
        pretrained_model_name_or_path_or_dict=path,
        weight_name=weight_name,
        use_safetensors=True,
        local_files_only=kwargs.pop("local_files_only", None),
        cache_dir=kwargs.pop("cache_dir", None),
        force_download=kwargs.pop("force_download", False),
        resume_download=kwargs.pop("resume_download", False),
        proxies=kwargs.pop("proxies", None),
        use_auth_token=kwargs.pop("use_auth_token", None),
        revision=kwargs.pop("revision", None),
        subfolder=kwargs.pop("subfolder", None),
        user_agent={"file_type": "attn_procs_weights", "framework": "pytorch"},
        allow_pickle=True,
    )

  @classmethod
  def load_lora_weights(cls, pretrained_model_name_or_path_or_dict, weight_name, adapter_name, weights_dtype, **kwargs):
    state_dict = cls.lora_state_dict(pretrained_model_name_or_path_or_dict, weight_name=weight_name, **kwargs)
    return convert_krea2_lora_to_flax(state_dict, adapter_name, weights_dtype)

  @classmethod
  def make_lora_interceptor(cls, ranks_by_path, network_alphas_by_path, adapter_name, scale=1.0):
    """Builds an `nn.intercept_methods` interceptor adding this adapter's LoRA
    update to every Dense whose module path appears in `ranks_by_path`."""
    lora_keys = frozenset(ranks_by_path.keys())

    def _intercept(next_fn, args, kwargs, context):
      mod = context.module
      while mod is not None:
        if isinstance(mod, BaseLoRALayer):
          return next_fn(*args, **kwargs)
        mod = mod.parent
      h = next_fn(*args, **kwargs)
      if context.method_name == "__call__":
        module_path = context.module.path
        if module_path in lora_keys:
          lora_layer = LoRALinearLayer(
              out_features=context.module.features,
              rank=ranks_by_path[module_path],
              network_alpha=network_alphas_by_path.get(module_path),
              dtype=context.module.dtype,
              weights_dtype=context.module.param_dtype,
              precision=context.module.precision,
              lora_scale=scale,
              name=f"lora-{adapter_name}",
          )
          return lora_layer(h, *args, **kwargs)
      return h

    return _intercept


def insert_lora_params(params, flat_lora_params):
  """Writes converted LoRA tensors into the (unfrozen) transformer param tree.

  Must run AFTER `load_and_convert_krea2_weights`, which zero-fills the lora-*
  leaves it doesn't recognize — writing earlier would silently zero the adapter.
  """
  if not flat_lora_params:
    return params
  flat = flatten_dict(params)
  for path, value in flat_lora_params.items():
    if path not in flat:
      raise ValueError(
          f"LoRA param {'.'.join(path)} is missing from the transformer param tree; "
          "the interceptors were not active while model shapes were evaluated."
      )
    if tuple(flat[path].shape) != tuple(value.shape):
      raise ValueError(
          f"LoRA param {'.'.join(path)} has shape {tuple(value.shape)} but the model "
          f"expects {tuple(flat[path].shape)}; this LoRA was likely trained for a different model."
      )
    flat[path] = value.astype(flat[path].dtype)
  return unflatten_dict(flat)


def apply_diff_updates(params, diff_updates, scale=1.0):
  """Additively merges `.diff` / `.diff_b` / norm-weight deltas into base params.
  Deltas are already in Flax layout (kernels transposed by the converter)."""
  if not diff_updates:
    return params
  flat = flatten_dict(params)
  for path, delta in diff_updates.items():
    if path not in flat:
      max_logging.log(f"WARNING: LoRA diff target {'.'.join(path)} not found in Krea 2 params; skipping.")
      continue
    leaf = flat[path]
    delta = jnp.asarray(delta).reshape(leaf.shape)
    flat[path] = (leaf + (delta * scale).astype(leaf.dtype)).astype(leaf.dtype)
  return unflatten_dict(flat)


def maybe_load_krea2_lora(config, weights_dtype=jnp.bfloat16):
  """Loads every adapter listed in `config.lora_config`.

  Returns `(flat_lora_params, interceptors, diff_updates)`. With no adapters
  configured, `flat_lora_params`/`diff_updates` are empty and `interceptors`
  holds a single no-op so callers can wrap unconditionally (Flux convention).
  """

  def _noop_interceptor(next_fn, args, kwargs, context):
    return next_fn(*args, **kwargs)

  lora_config = config.lora_config
  model_paths = lora_config["lora_model_name_or_path"]
  if len(model_paths) == 0:
    return {}, [_noop_interceptor], {}

  def _entry(key, i, default=None):
    values = lora_config.get(key, [])
    return values[i] if i < len(values) else default

  adapter_names = [_entry("adapter_name", i) or f"adapter_{i}" for i in range(len(model_paths))]
  assert len(set(adapter_names)) == len(adapter_names), f"Duplicate LoRA adapter names: {adapter_names}"

  flat_lora_params = {}
  interceptors = []
  diff_updates = {}
  for i, model_path in enumerate(model_paths):
    adapter_name = adapter_names[i]
    scale = float(_entry("scale", i, 1.0))
    max_logging.log(f"Loading Krea 2 LoRA '{adapter_name}' from {model_path} (scale={scale})...")
    lora_params, ranks, alphas, diffs = Krea2LoraLoaderMixin.load_lora_weights(
        model_path,
        weight_name=_entry("weight_name", i),
        adapter_name=adapter_name,
        weights_dtype=weights_dtype,
    )
    flat_lora_params.update(lora_params)
    for path, delta in diffs.items():
      scaled = jnp.asarray(delta) * scale
      diff_updates[path] = diff_updates[path] + scaled if path in diff_updates else scaled
    interceptors.append(Krea2LoraLoaderMixin.make_lora_interceptor(ranks, alphas, adapter_name, scale))
    max_logging.log(
        f"Krea 2 LoRA '{adapter_name}': {len(ranks)} LoRA module(s), {len(diffs)} diff update(s) loaded."
    )
  return flat_lora_params, interceptors, diff_updates

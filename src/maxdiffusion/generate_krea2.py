"""
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# End-to-end inference entry point for Krea 2 (K2) Raw and Turbo on JAX.
#
#   python src/maxdiffusion/generate_krea2.py src/maxdiffusion/configs/base_krea2.yml \
#     run_name=krea2_raw output_dir=output/ prompt="a fox in the snow"
#
#   python src/maxdiffusion/generate_krea2.py src/maxdiffusion/configs/base_krea2_turbo.yml \
#     run_name=krea2_turbo output_dir=output/ prompt="a fox in the snow"

import gc
import json
import math
import os
import time
from typing import List

from absl import app
import jax
import jax.numpy as jnp
import numpy as np
import flax
from flax import linen as nn
from flax import nnx
from flax.linen import partitioning as nn_partitioning
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from maxdiffusion import pyconfig
from maxdiffusion import max_logging
from maxdiffusion import max_utils
from maxdiffusion.max_utils import create_device_mesh


def partition_prompts(prompt_str: str, batch_size: int) -> List[str]:
  """Splits a prompt string by '||' and replicates/truncates to fill the batch_size."""
  raw_prompts = [p.strip() for p in prompt_str.split("||") if p.strip()]
  if not raw_prompts:
    raw_prompts = ["a fox in the snow"]

  num_prompts = len(raw_prompts)
  if num_prompts == 1:
    return raw_prompts * batch_size
  elif num_prompts <= batch_size:
    reps = batch_size // num_prompts
    active = []
    for p in raw_prompts:
      active.extend([p] * reps)
    if len(active) < batch_size:
      active.extend([raw_prompts[-1]] * (batch_size - len(active)))
    return active
  else:
    max_logging.log(
        f"Warning: Found {num_prompts} prompts, but batch_size is {batch_size}. Truncating to the first {batch_size}."
    )
    return raw_prompts[:batch_size]


def load_qwen_image_vae(snapshot_dir, config, vae_mesh, rngs):
  """Loads the Qwen-Image VAE (Wan 2.1 architecture) from the Krea 2 snapshot."""
  from maxdiffusion.models.wan.autoencoder_kl_wan import AutoencoderKLWan, AutoencoderKLWanCache
  from maxdiffusion.models.wan.wan_utils import load_wan_vae
  from functools import partial

  def create_model(rngs):
    return AutoencoderKLWan.from_config(
        snapshot_dir,
        subfolder="vae",
        rngs=rngs,
        mesh=vae_mesh,
        dtype=config.activations_dtype,
        weights_dtype=config.weights_dtype,
        vae_decode_chunk=1,
        vae_encode_chunk=4,
    )

  wan_vae = nnx.eval_shape(partial(create_model), rngs=rngs)
  graphdef, state = nnx.split(wan_vae, nnx.Param)
  params = state.to_pure_dict()
  state = dict(nnx.to_flat_state(state))

  params = load_wan_vae(snapshot_dir, params, "cpu")
  params = jax.tree_util.tree_map(lambda x: x.astype(config.weights_dtype), params)
  # The VAE is small; replicate it across the VAE mesh.
  replicated_sharding = NamedSharding(vae_mesh, P())
  for path, val in flax.traverse_util.flatten_dict(params).items():
    state[path].value = max_utils.device_put_replicated(val, replicated_sharding)
  state = nnx.from_flat_state(state)

  wan_vae = nnx.merge(graphdef, state)
  vae_cache = AutoencoderKLWanCache(wan_vae)
  return wan_vae, vae_cache


def main(argv):
  jax.config.update("jax_use_shardy_partitioner", True)

  # 1. Load configurations
  config_path = "src/maxdiffusion/configs/base_krea2.yml"
  custom_overrides = []
  if len(argv) > 1:
    if argv[1].endswith(".yml") or argv[1].endswith(".yaml"):
      config_path = argv[1]
      if len(argv) > 2:
        custom_overrides = argv[2:]
    else:
      custom_overrides = argv[1:]

  max_logging.log(f"Initializing pyconfig with config: {config_path}")
  default_args = [
      None,
      config_path,
      "run_name=krea2_generation",
      "output_dir=output/",
  ]
  default_args.extend(custom_overrides)
  pyconfig.initialize(default_args)

  # Import modules after jax.distributed.initialize() has run via pyconfig.initialize()
  from maxdiffusion.models.krea2.transformer_krea2_flax import Krea2Transformer2DModel
  from maxdiffusion.models.krea2.util import (
      KREA2_PROMPT_TEMPLATE_START_IDX,
      load_and_convert_krea2_weights,
      round_up_to_multiple,
  )
  from maxdiffusion.models.qwen3_flax import FlaxQwen3Config, FlaxQwen3Model, load_and_convert_qwen3_weights
  from maxdiffusion.models.flux.util import cast_dict_to_bfloat16_inplace
  from maxdiffusion.schedulers.scheduling_flow_match_flax import FlaxFlowMatchScheduler
  from maxdiffusion.pipelines.krea2.krea2_pipeline import FlaxKrea2Pipeline

  config = pyconfig.config
  os.makedirs(config.output_dir, exist_ok=True)

  # Resolve prompts early: the attention kernel choice below depends on whether
  # the batch mixes different prompts.
  active_prompts = partition_prompts(config.prompt, config.batch_size)

  # The repo's flash-attention kernels share the text padding mask of batch
  # element 0 across the whole batch. With mixed prompts in one batch that would
  # silently miscompute every other element, so fall back to dot_product.
  if config.attention != "dot_product" and config.batch_size > 1 and len(set(active_prompts)) > 1:
    max_logging.log(
        f"Warning: attention='{config.attention}' cannot honor per-batch text padding masks and the batch "
        "mixes different prompts. Falling back to attention='dot_product'. Use identical prompts per batch "
        "or batch_size=1 to keep flash attention."
    )
    pyconfig._config.keys["attention"] = "dot_product"

  # 2. Setup device meshes
  # The ICI parallelism product must equal the number of devices PER SLICE
  # (see max_utils.create_device_mesh), not the global device count.
  all_devices = jax.devices()
  try:
    num_slices = 1 + max(d.slice_index for d in all_devices)
  except Exception:
    num_slices = 1
  devices_per_slice = len(all_devices) // num_slices
  if config.batch_size == 1 and config.ici_tensor_parallelism == 1 and devices_per_slice > 1:
    max_logging.log(
        f"Auto-configuring Tensor Parallelism: ici_tensor_parallelism={devices_per_slice}, "
        f"ici_fsdp_parallelism=1 for batch_size=1 on {devices_per_slice} devices per slice "
        f"({num_slices} slice(s))."
    )
    pyconfig._config.keys["ici_tensor_parallelism"] = devices_per_slice
    pyconfig._config.keys["ici_fsdp_parallelism"] = 1

  max_logging.log("Setting up JAX device mesh...")
  devices_array = create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Dedicated mesh for the Wan-architecture VAE (axes: redundant, vae_spatial).
  vae_spatial = getattr(config, "vae_spatial", -1)
  total_devices = math.prod(devices_array.shape)
  if vae_spatial == -1:
    vae_spatial = total_devices
  assert (
      total_devices % vae_spatial == 0
  ), f"total devices ({total_devices}) must be a multiple of vae_spatial ({vae_spatial})"
  vae_devices_array = devices_array.flatten().reshape(total_devices // vae_spatial, vae_spatial)
  vae_mesh = Mesh(vae_devices_array, ("redundant", "vae_spatial"))
  vae_logical_axis_rules = getattr(config, "vae_logical_axis_rules", None)

  # 3. Resolve weights repository snapshot
  repo_id = config.pretrained_model_name_or_path
  max_logging.log(f"Target model: {repo_id}")
  if os.path.exists(repo_id):
    snapshot_dir = repo_id
    max_logging.log(f"Using local model directory: {snapshot_dir}")
  else:
    from huggingface_hub import snapshot_download

    max_logging.log(f"Resolving snapshot directory for model '{repo_id}' from HF Hub...")
    snapshot_dir = snapshot_download(repo_id=repo_id)

  max_logging.log(f"Host {jax.process_index()} using snapshot directory: {snapshot_dir}")
  transformer_path = os.path.join(snapshot_dir, "transformer")
  text_encoder_path = os.path.join(snapshot_dir, "text_encoder")
  tokenizer_path = os.path.join(snapshot_dir, "tokenizer")

  # 4. Text encoder config (Qwen3-VL: text tower lives under `text_config`)
  with open(os.path.join(text_encoder_path, "config.json"), "r") as f:
    te_config = json.load(f)
  text_config = te_config.get("text_config", te_config)
  rope_parameters = text_config.get("rope_parameters", {})
  rope_theta = rope_parameters.get("rope_theta", text_config.get("rope_theta", 5000000.0))

  qwen3_config = FlaxQwen3Config(
      vocab_size=text_config["vocab_size"],
      hidden_size=text_config["hidden_size"],
      intermediate_size=text_config["intermediate_size"],
      num_hidden_layers=text_config["num_hidden_layers"],
      num_attention_heads=text_config["num_attention_heads"],
      num_key_value_heads=text_config["num_key_value_heads"],
      head_dim=text_config.get("head_dim", 128),
      max_position_embeddings=text_config.get("max_position_embeddings", 262144),
      rms_norm_eps=text_config.get("rms_norm_eps", 1e-6),
      rope_theta=rope_theta,
      dtype=config.weights_dtype,
  )
  qwen3_model = FlaxQwen3Model(qwen3_config)

  # 5. Transformer config
  transformer_cfg = {}
  transformer_config_json = os.path.join(transformer_path, "config.json")
  if os.path.exists(transformer_config_json):
    with open(transformer_config_json, "r") as f:
      transformer_cfg = json.load(f)

  num_layers = transformer_cfg.get("num_layers", 28)
  transformer = Krea2Transformer2DModel(
      in_channels=transformer_cfg.get("in_channels", 64),
      num_layers=num_layers,
      attention_head_dim=transformer_cfg.get("attention_head_dim", 128),
      num_attention_heads=transformer_cfg.get("num_attention_heads", 48),
      num_key_value_heads=transformer_cfg.get("num_key_value_heads", 12),
      intermediate_size=transformer_cfg.get("intermediate_size", 16384),
      timestep_embed_dim=transformer_cfg.get("timestep_embed_dim", 256),
      text_hidden_dim=transformer_cfg.get("text_hidden_dim", 2560),
      num_text_layers=transformer_cfg.get("num_text_layers", 12),
      text_num_attention_heads=transformer_cfg.get("text_num_attention_heads", 20),
      text_num_key_value_heads=transformer_cfg.get("text_num_key_value_heads", 20),
      text_intermediate_size=transformer_cfg.get("text_intermediate_size", 6912),
      num_layerwise_text_blocks=transformer_cfg.get("num_layerwise_text_blocks", 2),
      num_refiner_text_blocks=transformer_cfg.get("num_refiner_text_blocks", 2),
      axes_dims_rope=tuple(transformer_cfg.get("axes_dims_rope", (32, 48, 48))),
      rope_theta=transformer_cfg.get("rope_theta", 1000.0),
      norm_eps=transformer_cfg.get("norm_eps", 1e-5),
      attention_kernel=config.attention,
      flash_block_sizes=max_utils.get_flash_block_sizes(config),
      mesh=mesh,
      dtype=config.activations_dtype,
      weights_dtype=config.weights_dtype,
  )

  # 6. Evaluate shapes & extract mesh shardings
  max_logging.log("Evaluating model shapes and shardings...")
  # Height/width must be multiples of 16 (VAE 8x downsampling x 2x2 latent
  # patches). Round up here so the eval_shape dummies match what the pipeline
  # (which applies the same rounding) will actually run.
  height = round_up_to_multiple(config.height, 16)
  width = round_up_to_multiple(config.width, 16)
  if (height, width) != (config.height, config.width):
    max_logging.log(
        f"Warning: height and width must be multiples of 16; rounding up from "
        f"{config.height}x{config.width} to {height}x{width}."
    )

  grid_h = height // 16
  grid_w = width // 16
  seq_len_img = grid_h * grid_w
  seq_len_txt = config.max_sequence_length
  # Total tokenized length before the system prefix is dropped.
  seq_len_txt_full = seq_len_txt + KREA2_PROMPT_TEMPLATE_START_IDX

  img_dummy = jnp.zeros((config.batch_size, seq_len_img, transformer_cfg.get("in_channels", 64)))
  txt_dummy = jnp.zeros(
      (
          config.batch_size,
          seq_len_txt,
          transformer_cfg.get("num_text_layers", 12),
          transformer_cfg.get("text_hidden_dim", 2560),
      )
  )
  t_dummy = jnp.zeros((config.batch_size,))
  img_ids_dummy = jnp.zeros((seq_len_img, 3))
  txt_ids_dummy = jnp.zeros((seq_len_txt, 3))
  text_mask_dummy = jnp.ones((config.batch_size, seq_len_txt), dtype=jnp.bool_)
  qwen_ids_dummy = jnp.zeros((config.batch_size, seq_len_txt_full), dtype=jnp.int32)
  qwen_mask_dummy = jnp.ones((config.batch_size, seq_len_txt_full), dtype=jnp.int32)

  key = jax.random.PRNGKey(config.seed if config.seed is not None else 0)
  key, qwen_key = jax.random.split(key)

  def transformer_init_fn():
    return transformer.init(
        key,
        hidden_states=img_dummy,
        encoder_hidden_states=txt_dummy,
        timestep=t_dummy,
        img_ids=img_ids_dummy,
        txt_ids=txt_ids_dummy,
        encoder_attention_mask=text_mask_dummy,
    )

  def qwen3_init_fn():
    return qwen3_model.init(qwen_key, qwen_ids_dummy, qwen_mask_dummy)

  with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
    abstract_transformer_vars = jax.eval_shape(transformer_init_fn)
    abstract_qwen3_vars = jax.eval_shape(qwen3_init_fn)

    logical_transformer_specs = nn.get_partition_spec(abstract_transformer_vars)
    logical_qwen3_specs = nn.get_partition_spec(abstract_qwen3_vars)

    transformer_mesh_shardings = nn.logical_to_mesh_sharding(logical_transformer_specs, mesh, config.logical_axis_rules)
    qwen3_mesh_shardings = nn.logical_to_mesh_sharding(logical_qwen3_specs, mesh, config.logical_axis_rules)

  transformer_shardings = flax.core.freeze(transformer_mesh_shardings["params"])
  qwen3_shardings = flax.core.freeze(qwen3_mesh_shardings["params"])

  # 7. Load weights on host CPU, then place on devices
  max_logging.log("Loading parameters on host CPU...")
  t_load_start = time.time()
  cpu_device = jax.local_devices(backend="cpu")[0]
  with jax.default_device(cpu_device):
    with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
      import flax.linen.spmd as flax_spmd

      def unbox_fn(x):
        return x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x

      params = jax.tree_util.tree_map(
          unbox_fn, abstract_transformer_vars["params"], is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned)
      )
      params = flax.core.unfreeze(params)

      qwen3_params = jax.tree_util.tree_map(
          unbox_fn, abstract_qwen3_vars["params"], is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned)
      )
      qwen3_params = flax.core.unfreeze(qwen3_params)

      params = load_and_convert_krea2_weights(transformer_path, params, num_layers)
      qwen3_params = load_and_convert_qwen3_weights(
          text_encoder_path, qwen3_params, qwen3_config, key_prefix="model.language_model."
      )

      if config.weights_dtype == jnp.bfloat16:
        max_logging.log("Casting Qwen3 parameters to bfloat16 (keeping norms in float32)...")
        cast_dict_to_bfloat16_inplace(qwen3_params, exclude_keywords=("norm",))

      params = flax.core.freeze(params)
      qwen3_params = flax.core.freeze(qwen3_params)

      max_logging.log("Placing parameters on device HBM...")
      with mesh, nn_partitioning.axis_rules(config.logical_axis_rules):
        params = jax.tree_util.tree_map(max_utils.device_put_replicated, params, transformer_shardings)
        qwen3_params = jax.tree_util.tree_map(max_utils.device_put_replicated, qwen3_params, qwen3_shardings)
      max_logging.log("All parameters placed on device HBM successfully!")
      gc.collect()
      jax.effects_barrier()

  # 8. VAE (Qwen-Image / Wan 2.1 architecture, NNX)
  max_logging.log("Loading Qwen-Image VAE...")
  rngs = nnx.Rngs(jax.random.key(config.seed if config.seed is not None else 0))
  vae, vae_cache = load_qwen_image_vae(snapshot_dir, config, vae_mesh, rngs)

  load_time = time.time() - t_load_start
  max_logging.log(f" -> [TIMING] Total Model Loading & Device Placement: {load_time:.2f} seconds")

  # 9. Tokenizer
  from transformers import AutoTokenizer

  try:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
  except Exception:
    tokenizer = AutoTokenizer.from_pretrained(snapshot_dir, subfolder="tokenizer", local_files_only=True)

  # 10. FlowMatch scheduler (exponential dynamic shifting; mu is set per-call)
  scheduler = FlaxFlowMatchScheduler(
      num_train_timesteps=1000,
      shift=1.0,
      sigma_max=1.0,
      sigma_min=0.001,
      inverse_timesteps=False,
      extra_one_step=False,
      reverse_sigmas=False,
      use_dynamic_shifting=True,
      time_shift_type="exponential",
  )

  # 11. Pipeline
  max_logging.log("Instantiating FlaxKrea2Pipeline...")
  pipeline = FlaxKrea2Pipeline(
      transformer=transformer,
      vae=vae,
      vae_cache=vae_cache,
      text_encoder=qwen3_model,
      tokenizer=tokenizer,
      scheduler=scheduler,
      config=config,
      mesh=mesh,
      vae_mesh=vae_mesh,
      vae_logical_axis_rules=vae_logical_axis_rules,
  )

  latents_to_use = None
  if getattr(config, "latents_path", ""):
    max_logging.log(f"Loading custom starting noise latents from: {config.latents_path}...")
    latents_to_use = np.load(config.latents_path)
    max_logging.log(f" -> Custom latents shape: {latents_to_use.shape}")

  call_kwargs = dict(
      params=params,
      qwen3_params=qwen3_params,
      height=height,
      width=width,
      num_inference_steps=config.num_inference_steps,
      guidance_scale=config.guidance_scale,
      negative_prompt=config.negative_prompt,
      batch_size=config.batch_size,
      latents=latents_to_use,
      output_dir=config.output_dir,
  )

  max_logging.log("Running warmup pass (XLA compilation)...")
  _, warmup_trace = pipeline(prompt=active_prompts, output_name="krea2_warmup.png", **call_kwargs)
  warmup_time = sum(warmup_trace.get(k, 0.0) for k in ("prompt_encoding", "denoise_loop", "vae_decode"))

  max_logging.log("Running timed pass at full device speed...")
  _, main_trace = pipeline(prompt=active_prompts, output_name=config.output_name, **call_kwargs)
  main_time = sum(main_trace.get(k, 0.0) for k in ("prompt_encoding", "denoise_loop", "vae_decode"))

  max_logging.log("=" * 80)
  max_logging.log("KREA 2 LATENCY & TIMING BREAKDOWN")
  max_logging.log("=" * 80)
  max_logging.log(f"1) Total Model Loading & Placement Time:  {load_time:.2f} seconds")
  max_logging.log(f"2) Cold-Start / Warmup Pass (XLA Compilation): {warmup_time:.2f} seconds")
  max_logging.log(f"   - Qwen3-VL Encoding: {warmup_trace.get('prompt_encoding', 0.0):.2f}s")
  max_logging.log(f"   - Krea2 Denoising:   {warmup_trace.get('denoise_loop', 0.0):.2f}s")
  max_logging.log(f"   - VAE Decoding:      {warmup_trace.get('vae_decode', 0.0):.2f}s")
  max_logging.log(f"3) Main Warmed-Up Pass: {main_time:.2f} seconds")
  max_logging.log(f"   - Qwen3-VL Encoding: {main_trace.get('prompt_encoding', 0.0):.2f}s")
  max_logging.log(f"   - Krea2 Denoising:   {main_trace.get('denoise_loop', 0.0):.2f}s")
  max_logging.log(f"   - VAE Decoding:      {main_trace.get('vae_decode', 0.0):.2f}s")
  max_logging.log("=" * 80)
  max_logging.log(f"SUCCESS! Generation complete for {config.batch_size} image(s)!")


if __name__ == "__main__":
  app.run(main)

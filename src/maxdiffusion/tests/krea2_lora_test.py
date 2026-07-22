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

# CPU-runnable unit tests for Krea 2 inference-time LoRA loading.

import unittest

import flax
import flax.linen as nn
import pytest
import flax.linen.spmd as flax_spmd
import jax
import jax.numpy as jnp
import numpy as np
from flax.traverse_util import flatten_dict, unflatten_dict

from maxdiffusion.loaders.krea2_lora_pipeline import (
    Krea2LoraLoaderMixin,
    apply_diff_updates,
    insert_lora_params,
)
from maxdiffusion.models.krea2.lora_util import (
    convert_krea2_lora_to_flax,
    krea2_torch_path_to_flax_path,
    normalize_krea2_lora_state_dict,
)
from maxdiffusion.models.krea2.transformer_krea2_flax import Krea2Transformer2DModel
from maxdiffusion.models.krea2.util import prepare_krea2_image_ids, prepare_krea2_text_ids


def _unbox(params):
  return jax.tree_util.tree_map(
      lambda x: x.unbox() if isinstance(x, flax_spmd.LogicallyPartitioned) else x,
      params,
      is_leaf=lambda k: isinstance(k, flax_spmd.LogicallyPartitioned),
  )


def _tiny_model(**overrides):
  kwargs = dict(
      in_channels=16,
      num_layers=2,
      attention_head_dim=8,
      num_attention_heads=4,  # hidden_size = 32
      num_key_value_heads=2,
      intermediate_size=64,
      timestep_embed_dim=16,
      text_hidden_dim=24,
      num_text_layers=3,
      text_num_attention_heads=4,
      text_num_key_value_heads=4,
      text_intermediate_size=48,
      num_layerwise_text_blocks=2,
      num_refiner_text_blocks=2,
      axes_dims_rope=(4, 2, 2),
  )
  kwargs.update(overrides)
  return Krea2Transformer2DModel(**kwargs)


def _tiny_inputs(batch_size=1, s_img=4, s_txt=3):
  hs = jnp.array(np.random.RandomState(0).randn(batch_size, s_img, 16), dtype=jnp.float32)
  ehs = jnp.array(np.random.RandomState(1).randn(batch_size, s_txt, 3, 24), dtype=jnp.float32)
  t = jnp.full((batch_size,), 0.5)
  img_ids = prepare_krea2_image_ids(batch_size, 2, 2)
  txt_ids = prepare_krea2_text_ids(batch_size, s_txt)
  mask = jnp.ones((batch_size, s_txt), dtype=jnp.bool_)
  return hs, ehs, t, img_ids, txt_ids, mask


def _rand(rng, *shape):
  return rng.randn(*shape).astype(np.float32)


class NormalizeStateDictTest(unittest.TestCase):
  """Key normalization across kohya / ComfyUI / diffusers-PEFT formats."""

  def test_all_three_formats_and_skips(self):
    rng = np.random.RandomState(0)
    state_dict = {
        # kohya underscore style + alpha
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": _rand(rng, 2, 32),
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": _rand(rng, 32, 2),
        "lora_unet_transformer_blocks_0_attn_to_q.alpha": np.float32(4.0),
        # ComfyUI dotted style, diffusers `ff.gate` naming
        "diffusion_model.transformer_blocks.1.ff.gate.lora_down.weight": _rand(rng, 2, 32),
        "diffusion_model.transformer_blocks.1.ff.gate.lora_up.weight": _rand(rng, 64, 2),
        # diffusers/PEFT style with to_out.0
        "transformer.transformer_blocks.1.attn.to_out.0.lora_A.weight": _rand(rng, 2, 32),
        "transformer.transformer_blocks.1.attn.to_out.0.lora_B.weight": _rand(rng, 32, 2),
        # text-encoder key -> skipped
        "lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight": _rand(rng, 2, 8),
        # unknown module -> unmatched
        "lora_unet_bogus_module.lora_down.weight": _rand(rng, 2, 8),
        # unknown suffix -> unmatched
        "some_random_tensor": _rand(rng, 3),
    }
    modules, skipped_te, unmatched = normalize_krea2_lora_state_dict(state_dict)

    entry = modules["transformer_blocks.0.attn.to_q"]
    self.assertEqual(entry["down"].shape, (2, 32))
    self.assertEqual(entry["up"].shape, (32, 2))
    self.assertEqual(entry["alpha"], 4.0)
    self.assertIn("transformer_blocks.1.ff.gate", modules)
    self.assertIn("transformer_blocks.1.attn.to_out.0", modules)
    self.assertEqual(skipped_te, ["lora_te_text_model_encoder_layers_0_self_attn_q_proj.lora_down.weight"])
    self.assertEqual(sorted(unmatched), ["lora_unet_bogus_module.lora_down.weight", "some_random_tensor"])

  def test_diff_and_diff_b_keys(self):
    rng = np.random.RandomState(0)
    state_dict = {
        "lora_unet_transformer_blocks_0_norm1.diff": _rand(rng, 32),
        "lora_unet_transformer_blocks_0_scale_shift_table.diff": _rand(rng, 6, 32),
        "lora_unet_img_in.diff_b": _rand(rng, 32),
    }
    modules, _, unmatched = normalize_krea2_lora_state_dict(state_dict)
    self.assertEqual(unmatched, [])
    self.assertIn("diff", modules["transformer_blocks.0.norm1"])
    self.assertIn("diff", modules["transformer_blocks.0.scale_shift_table"])
    self.assertIn("diff_b", modules["img_in"])

  def test_fused_qkv_raises(self):
    state_dict = {"lora_unet_transformer_blocks_0_attn_qkv.lora_down.weight": np.zeros((2, 32), np.float32)}
    with self.assertRaises(NotImplementedError):
      normalize_krea2_lora_state_dict(state_dict)

  def test_torch_tensors_are_converted(self):
    torch = pytest.importorskip("torch")

    state_dict = {
        "lora_unet_img_in.lora_down.weight": torch.randn(2, 16, dtype=torch.bfloat16),
        "lora_unet_img_in.lora_up.weight": torch.randn(32, 2, dtype=torch.bfloat16),
        "lora_unet_img_in.alpha": torch.tensor(2.0),
    }
    modules, _, unmatched = normalize_krea2_lora_state_dict(state_dict)
    self.assertEqual(unmatched, [])
    self.assertIsInstance(modules["img_in"]["down"], np.ndarray)
    self.assertEqual(modules["img_in"]["down"].dtype, np.float32)
    self.assertEqual(modules["img_in"]["alpha"], 2.0)


class TorchPathToFlaxPathTest(unittest.TestCase):

  def test_mappings(self):
    cases = {
        "transformer_blocks.3.attn.to_q": ("blocks_3", "attn", "to_q"),
        "transformer_blocks.3.attn.to_out.0": ("blocks_3", "attn", "to_out"),
        "transformer_blocks.3.attn.to_out": ("blocks_3", "attn", "to_out"),
        "transformer_blocks.3.ff.gate": ("blocks_3", "ff", "gate_proj"),
        "transformer_blocks.3.ff.down_proj": ("blocks_3", "ff", "down_proj"),
        "transformer_blocks.3.norm2": ("blocks_3", "norm2"),
        "transformer_blocks.3.scale_shift_table": ("blocks_3", "scale_shift_table"),
        "text_fusion.layerwise_blocks.1.attn.to_v": ("text_fusion", "layerwise_blocks_1", "attn", "to_v"),
        "text_fusion.refiner_blocks.0.ff.up": ("text_fusion", "refiner_blocks_0", "ff", "up_proj"),
        "text_fusion.projector": ("text_fusion", "projector"),
        "img_in": ("img_in",),
        "time_mod_proj": ("time_mod_proj",),
        "time_embed.linear_1": ("time_embed", "linear_1"),
        "txt_in.linear_2": ("txt_in", "linear_2"),
        "txt_in.norm": ("txt_in", "norm"),
        "final_layer.linear": ("final_layer", "linear"),
        "final_layer.scale_shift_table": ("final_layer", "scale_shift_table"),
    }
    for torch_path, expected in cases.items():
      self.assertEqual(krea2_torch_path_to_flax_path(torch_path), expected, torch_path)

  def test_invalid_paths(self):
    for bad in (
        "bogus_module",
        "transformer_blocks.x.attn.to_q",
        "transformer_blocks.0.attn.to_w",
        "time_embed.norm",
        "text_fusion.other_blocks.0.attn.to_q",
    ):
      self.assertIsNone(krea2_torch_path_to_flax_path(bad), bad)


class ConverterTest(unittest.TestCase):

  def _kohya_state_dict(self):
    rng = np.random.RandomState(7)
    return {
        # blocks_0.attn.to_q: hidden 32 -> 32
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": _rand(rng, 2, 32),
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": _rand(rng, 32, 2),
        "lora_unet_transformer_blocks_0_attn_to_q.alpha": np.float32(4.0),
        # blocks_1.ff.down_proj: 64 -> 32 (ComfyUI dotted)
        "diffusion_model.transformer_blocks.1.ff.down.lora_down.weight": _rand(rng, 2, 64),
        "diffusion_model.transformer_blocks.1.ff.down.lora_up.weight": _rand(rng, 32, 2),
        # text_fusion refiner attn.to_k: 24 -> 24 (kv heads 4 * head_dim 6)
        "lora_unet_text_fusion_refiner_blocks_0_attn_to_k.lora_down.weight": _rand(rng, 2, 24),
        "lora_unet_text_fusion_refiner_blocks_0_attn_to_k.lora_up.weight": _rand(rng, 24, 2),
        # img_in: 16 -> 32 (PEFT keys, no alpha)
        "transformer.img_in.lora_A.weight": _rand(rng, 2, 16),
        "transformer.img_in.lora_B.weight": _rand(rng, 32, 2),
        # diffs
        "lora_unet_transformer_blocks_0_norm1.diff": _rand(rng, 32),
        "lora_unet_transformer_blocks_0_scale_shift_table.diff": _rand(rng, 6, 32),
        "lora_unet_img_in.diff_b": _rand(rng, 32),
        # full-weight diff on a Dense (torch layout (out, in) -> transposed)
        "lora_unet_final_layer_linear.diff": _rand(rng, 16, 32),
    }

  def test_converted_paths_match_model_tree(self):
    state_dict = self._kohya_state_dict()
    flat_lora, ranks, alphas, diffs = convert_krea2_lora_to_flax(state_dict, "test", weights_dtype=jnp.float32)

    model = _tiny_model()
    base_params = _unbox(model.init(jax.random.PRNGKey(0), *_tiny_inputs())["params"])
    flat_base = flatten_dict(flax.core.unfreeze(base_params))

    self.assertEqual(len(ranks), 4)
    for module_path, rank in ranks.items():
      base_kernel = flat_base[(*module_path, "kernel")]
      down = flat_lora[(*module_path, "lora-test", "down", "kernel")]
      up = flat_lora[(*module_path, "lora-test", "up", "kernel")]
      # down: (in, r), up: (r, out) against the base (in, out) kernel.
      self.assertEqual(down.shape, (base_kernel.shape[0], rank))
      self.assertEqual(up.shape, (rank, base_kernel.shape[1]))

    self.assertEqual(alphas[("blocks_0", "attn", "to_q")], 4.0)
    self.assertIsNone(alphas[("img_in",)])

    # Values are transposed torch tensors.
    np.testing.assert_allclose(
        np.asarray(flat_lora[("blocks_0", "attn", "to_q", "lora-test", "down", "kernel")]),
        state_dict["lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight"].T,
    )

    # Diff targets exist in the base tree with matching shapes (kernel diffs transposed).
    for path, delta in diffs.items():
      self.assertIn(path, flat_base, path)
      self.assertEqual(tuple(np.asarray(delta).shape), tuple(flat_base[path].shape), path)

  def test_rank_larger_than_out_features_is_skipped(self):
    rng = np.random.RandomState(0)
    state_dict = {
        # projector maps 3 -> 1; rank 2 > out_features 1 must be skipped.
        "lora_unet_text_fusion_projector.lora_down.weight": _rand(rng, 2, 3),
        "lora_unet_text_fusion_projector.lora_up.weight": _rand(rng, 1, 2),
    }
    flat_lora, ranks, _, _ = convert_krea2_lora_to_flax(state_dict, "test")
    self.assertEqual(flat_lora, {})
    self.assertEqual(ranks, {})


class _DenseHost(nn.Module):
  """Minimal host so the intercepted Dense has a non-empty module path."""

  features: int = 8

  @nn.compact
  def __call__(self, x):
    return nn.Dense(self.features, use_bias=False, name="proj")(x)


class InterceptorTest(unittest.TestCase):

  def test_interceptor_matches_manual_lora_math(self):
    rank, alpha, scale = 2, 4.0, 0.7
    host = _DenseHost()
    x = jnp.array(np.random.RandomState(0).randn(3, 5), dtype=jnp.float32)

    interceptor = Krea2LoraLoaderMixin.make_lora_interceptor(
        {("proj",): rank}, {("proj",): alpha}, "test", scale=scale
    )
    with nn.intercept_methods(interceptor):
      params = flax.core.unfreeze(host.init(jax.random.PRNGKey(0), x)["params"])

    rng = np.random.RandomState(1)
    flat = flatten_dict(params)
    self.assertIn(("proj", "lora-test", "down", "kernel"), flat)
    down = _rand(rng, 5, rank)
    up = _rand(rng, rank, 8)
    flat[("proj", "lora-test", "down", "kernel")] = jnp.asarray(down)
    flat[("proj", "lora-test", "up", "kernel")] = jnp.asarray(up)
    params = unflatten_dict(flat)

    with nn.intercept_methods(interceptor):
      out = host.apply({"params": params}, x)

    kernel = np.asarray(flat[("proj", "kernel")])
    expected = np.asarray(x) @ kernel + scale * (alpha / rank) * ((np.asarray(x) @ down) @ up)
    np.testing.assert_allclose(np.asarray(out), expected, rtol=1e-5, atol=1e-5)

  def test_zero_scale_is_inert(self):
    host = _DenseHost()
    x = jnp.array(np.random.RandomState(0).randn(3, 5), dtype=jnp.float32)
    interceptor = Krea2LoraLoaderMixin.make_lora_interceptor({("proj",): 2}, {("proj",): None}, "test", scale=0.0)
    with nn.intercept_methods(interceptor):
      params = flax.core.unfreeze(host.init(jax.random.PRNGKey(0), x)["params"])
    flat = flatten_dict(params)
    flat[("proj", "lora-test", "down", "kernel")] = jnp.ones((5, 2))
    flat[("proj", "lora-test", "up", "kernel")] = jnp.ones((2, 8))
    params = unflatten_dict(flat)

    with nn.intercept_methods(interceptor):
      out_lora = host.apply({"params": params}, x)
    base_out = host.apply({"params": {"proj": {"kernel": params["proj"]["kernel"]}}}, x)
    np.testing.assert_array_equal(np.asarray(out_lora), np.asarray(base_out))


class EndToEndTinyModelTest(unittest.TestCase):

  def _lora_state_dict(self, rng):
    return {
        "lora_unet_transformer_blocks_0_attn_to_q.lora_down.weight": _rand(rng, 2, 32),
        "lora_unet_transformer_blocks_0_attn_to_q.lora_up.weight": _rand(rng, 32, 2),
        "lora_unet_transformer_blocks_0_attn_to_q.alpha": np.float32(2.0),
    }

  def test_lora_changes_model_output(self):
    model = _tiny_model()
    inputs = _tiny_inputs()
    flat_lora, ranks, alphas, _ = convert_krea2_lora_to_flax(
        self._lora_state_dict(np.random.RandomState(3)), "style", weights_dtype=jnp.float32
    )
    interceptor = Krea2LoraLoaderMixin.make_lora_interceptor(ranks, alphas, "style", scale=1.0)

    with nn.intercept_methods(interceptor):
      params = flax.core.unfreeze(_unbox(model.init(jax.random.PRNGKey(0), *inputs)["params"]))
    params = insert_lora_params(params, flat_lora)

    base_out = model.apply({"params": params}, *inputs).sample
    with nn.intercept_methods(interceptor):
      lora_out = model.apply({"params": params}, *inputs).sample

    self.assertEqual(base_out.shape, lora_out.shape)
    self.assertTrue(np.any(np.abs(np.asarray(base_out) - np.asarray(lora_out)) > 1e-6))

  def test_eval_shape_tree_contains_lora_leaves(self):
    """Regression test for the sharding/param tree mismatch: the abstract tree
    produced under the interceptor must contain every converted LoRA leaf."""
    model = _tiny_model()
    inputs = _tiny_inputs()
    flat_lora, ranks, alphas, _ = convert_krea2_lora_to_flax(
        self._lora_state_dict(np.random.RandomState(3)), "style", weights_dtype=jnp.float32
    )
    interceptor = Krea2LoraLoaderMixin.make_lora_interceptor(ranks, alphas, "style", scale=1.0)

    with nn.intercept_methods(interceptor):
      abstract_vars = jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), *inputs))

    flat_abstract = flatten_dict(flax.core.unfreeze(_unbox(abstract_vars["params"])))
    for path, value in flat_lora.items():
      self.assertIn(path, flat_abstract, path)
      self.assertEqual(tuple(flat_abstract[path].shape), tuple(value.shape), path)

  def test_insert_lora_params_rejects_unknown_or_mismatched(self):
    model = _tiny_model()
    inputs = _tiny_inputs()
    params = flax.core.unfreeze(_unbox(model.init(jax.random.PRNGKey(0), *inputs)["params"]))
    with self.assertRaises(ValueError):
      insert_lora_params(params, {("blocks_0", "attn", "to_q", "lora-x", "down", "kernel"): jnp.zeros((32, 2))})


class DiffMergeTest(unittest.TestCase):

  def test_diff_and_diff_b_merge_with_scale(self):
    rng = np.random.RandomState(0)
    state_dict = {
        "lora_unet_transformer_blocks_0_norm1.diff": _rand(rng, 32),
        "lora_unet_final_layer_linear.diff": _rand(rng, 16, 32),
        "lora_unet_final_layer_linear.diff_b": _rand(rng, 16),
    }
    _, _, _, diffs = convert_krea2_lora_to_flax(state_dict, "test")

    params = {
        "blocks_0": {"norm1": {"weight": jnp.zeros((32,), jnp.float32)}},
        "final_layer": {"linear": {"kernel": jnp.ones((32, 16), jnp.float32), "bias": jnp.zeros((16,), jnp.float32)}},
    }
    merged = apply_diff_updates(params, diffs, scale=0.5)

    np.testing.assert_allclose(
        np.asarray(merged["blocks_0"]["norm1"]["weight"]),
        0.5 * state_dict["lora_unet_transformer_blocks_0_norm1.diff"],
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(merged["final_layer"]["linear"]["kernel"]),
        1.0 + 0.5 * state_dict["lora_unet_final_layer_linear.diff"].T,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        np.asarray(merged["final_layer"]["linear"]["bias"]),
        0.5 * state_dict["lora_unet_final_layer_linear.diff_b"],
        rtol=1e-6,
    )


if __name__ == "__main__":
  unittest.main()

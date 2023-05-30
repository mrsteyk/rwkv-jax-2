import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax

from functools import partial
from einops import rearrange

import transformer_block

class RWKV(hk.Module):
    def __init__(self, config):
        super().__init__()
        self._vocab_size = config["vocab_size"]
        self._n_embd = config["n_embd"]
        self._config = config
        self._n_layers = config["n_layers"]
        self._batch_first = config["batch_first"]
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        x = hk.Embed(self._vocab_size, self._n_embd, w_init=embed_init, name="emb")(x)

        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="ln0")(x)
        hiddens = x.shape[-1]

        # print(x.shape)
        # blocks = [transformer_block.create_block(layer_id, layers, config) for layer_id in range(self._n_layers)]
        for i in range(self._n_layers):
            # for J3 and mine
            # x1, x2 = jnp.split(
            #     # TODO: config value?
            #     # hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f'l{i}_ln')(x),
            #     x,
            #     [hiddens // 2], axis=-1
            # )
            # # print(x1.shape, x2.shape)
            f, g = transformer_block.create_block(i, self._n_layers, self._config)
            # J3
            # y1 = f(x2) + x1
            # y2 = g(y1) + x2
            # assert y1.shape == x1.shape
            # assert y2.shape == x2.shape
            # x = jnp.concatenate([y1, y2], axis=-1)
            # # Mine, weird interpret of J3
            # x = jnp.concatenate([x1 + f(x1), x2 + g(x2)], axis=-1)

            # For J2 and 1? Fun fact: NeoX codebase had a bug where it wasn't actually tied J residuals! Behind config var for backcompat with 20b
            # Source: https://github.com/EleutherAI/gpt-neox/blob/335514210ad5226637cce647d6251d26819ca147/megatron/model/transformer.py#L805-L810
            hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f'l{i}_ln')(x)
            # J1 tied ln
            x = x + f(x) + g(x)
            
            # J2 does something funny akin to that?
            # BUT hardcoded widening factor for MLP in J2 is 8, I use 4 as per J3
            # xx = jnp.concatenate([f(x), g(x)], axis=-1)
            # x = x + hk.Linear(hiddens, w_init=hk.initializers.VarianceScaling(2 / self._n_layers), with_bias=False, name=f"l{i}_o")(xx)
        
        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="ln_out")(x)
        x = hk.Linear(self._vocab_size, name="lm_head")(x)
        # if not self._batch_first:
        #     x = rearrange(x, "s b e -> b s e")
        return x

def loss_fn(apply_fn, weights):
    y_pred = apply_fn(weights, None, x)
    return optax.softmax_cross_entropy_with_integer_labels(y_pred, y).mean()

def loss_fn_grad(apply_fn):
    def f(x, y, weights):
        y_pred = apply_fn(weights, None, x)
        return optax.softmax_cross_entropy_with_integer_labels(y_pred, y).mean()
    return jax.value_and_grad(f, argnums=2, allow_int=True)

def RWKV_CharLevel(mlp="rwkv", batch_first = False, do_rearrange = True):
    config = {
        "batch_first": batch_first,
        "do_rearrange": do_rearrange,

        "vocab_size": 256,
        "n_embd": 512,
        "n_layers": 6,

        "mlp": {
            "layer_name": mlp,
            "widening_factor": 4,
            "bias": False,
        },
        "attn": {
            "widening_factor": 1,
        },
    }
    # def f(x):
    #     rwkv = RWKV(config)
    #     return rwkv(x)
    return hk.transform(lambda x: RWKV(config)(x)), config
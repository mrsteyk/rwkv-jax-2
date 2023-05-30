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
        self._j_residuals = (config["j_residual"],config["j2_residual"],config["j3_residual"])
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        embed_init = hk.initializers.TruncatedNormal(stddev=0.02)
        x = hk.Embed(self._vocab_size, self._n_embd, w_init=embed_init, name="emb")(x)

        x = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name="ln0")(x)
        hiddens = x.shape[-1]

        # print(x.shape)
        # blocks = [transformer_block.create_block(layer_id, layers, config) for layer_id in range(self._n_layers)]
        for i in range(self._n_layers):
            f, g = transformer_block.create_block(i, self._n_layers, self._config)
            if self._j_residuals == (False, False, False) or self._j_residuals == (False, False, True):
                # for J3 and mine
                x1, x2 = jnp.split(
                    # TODO: config value?
                    # hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f'l{i}_ln')(x),
                    x,
                    [hiddens // 2], axis=-1
                )
                if self._j_residuals[2]:
                    # J3
                    y1 = f(x2) + x1
                    y2 = g(y1) + x2
                    assert y1.shape == x1.shape
                    assert y2.shape == x2.shape
                    x = jnp.concatenate([y1, y2], axis=-1)
                else:
                    # Mine, weird interpret of J3
                    x = jnp.concatenate([x1 + f(x1), x2 + g(x2)], axis=-1)
            elif self._j_residuals[0] or self._j_residuals[1]:
                # For J2 and 1? Fun fact: NeoX codebase had a bug where it wasn't actually tied J residuals! Behind config var for backcompat with 20b
                # Source: https://github.com/EleutherAI/gpt-neox/blob/335514210ad5226637cce647d6251d26819ca147/megatron/model/transformer.py#L805-L810
                xx = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True, name=f'l{i}_ln')(x)
                # J1 tied ln
                if self._j_residuals[0]:
                    x = x + f(xx) + g(xx)
                
                # J2 does something funny akin to that?
                # BUT hardcoded widening factor for MLP in J2 is 8, I use 4 as per J3
                # I think it's cuz of GLU? //2 to g*gelu(o)
                if self._j_residuals[1]:
                    # f should be attn
                    # g should be mlp
                    xx = jnp.concatenate([f(xx), g(xx)], axis=-1)
                    # Corrrect one, yields wierd loss only for linear MLP, better for everyone else!
                    # I suspect it's because of gelu and not glu in linear MLP?
                    j2_init = hk.initializers.TruncatedNormal(stddev=1.0 / jnp.sqrt(hiddens))
                    # j2_init = hk.initializers.VarianceScaling(2.0 / self._n_layers)
                    x = x + hk.Linear(hiddens, w_init=j2_init, with_bias=False, name=f"l{i}_o")(xx)
            else:
                raise
        
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

def RWKV_CharLevel(mlp="rwkv", batch_first = False, do_rearrange = True, j_residual=False, j2_residual=True, j3_residual=False):
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
        "j_residual": j_residual,
        "j2_residual": j2_residual,
        "j3_residual": j3_residual,
    }
    # def f(x):
    #     rwkv = RWKV(config)
    #     return rwkv(x)
    return hk.transform(lambda x: RWKV(config)(x)), config
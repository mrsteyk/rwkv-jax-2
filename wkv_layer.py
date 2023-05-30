import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from typing import Optional, Callable

from functools import partial
from einops import repeat, rearrange

def exp_mix_frac(p1, p2, v1_upper, v1_lower, v2_upper, v2_lower):
    p = jnp.maximum(p1, p2)
    e1 = jnp.exp(p1 - p)
    e2 = jnp.exp(p2 - p)
    return v1_upper * e1 + v2_upper * e2, v1_lower * e1 + v2_lower * e2, p

def assoc_reduce_step(left, right):
    (expkv_l, expk_l, w_l, p_l) = left
    (expkv_r, expk_r, w_r, p_r) = right
    a, b, p = exp_mix_frac(p_l + w_r, p_r, expkv_l, expk_l, expkv_r, expk_r)
    return a, b, w_l + w_r, p

def time_mix(x, x_prev, mix):
    return mix * x + (1 - mix) * x_prev

@partial(jax.jit, static_argnums=4)
def wkv_single_channel(wc, uc, kc, vc, state_in_c=(0,0,-1e38)):
    wc = -jnp.exp(wc)
    # @jax.checkpoint
    def step(pqo, kct_vct):
        kct, vct = kct_vct

        p, q, o = pqo
        no = jnp.maximum(o, uc + kct).astype(kct.dtype)
        A = jnp.exp(o - no)
        B = jnp.exp(uc + kct - no)
        y = (A * p + B * vct) / (A * q + B)

        no = jnp.maximum(wc + o, kct)
        A = jnp.exp(wc + o - no)
        B = jnp.exp(kct - no)
        p = A * p + B * vct
        q = A * q + B
        o = no

        return (p, q, o), y

    state_out_c, y = jax.lax.scan(step, state_in_c, (kc, vc))
    return y, state_out_c

WKV_gpu = jax.vmap(wkv_single_channel, -1, -1)

WKV_gpu_batch = jax.vmap(WKV_gpu, (None, None, 0, 0, None))

@hk.transparent
def rkv(x, x_prev, time_mix_init, init_scale, widening_factor = 1):
    hiddens = x.shape[-1]
    initializer = hk.initializers.VarianceScaling(init_scale)

    # print(x.shape)
    # is it even better?
    tmrkv = hk.get_parameter("time_mix_rkv", (hiddens * 3,), init=time_mix_init)
    x = jnp.concatenate([x,x,x], axis=-1)
    x_prev = jnp.concatenate([x_prev,x_prev,x_prev], axis=-1)
    x_r, x_k, x_v = jnp.split(time_mix(x, x_prev, tmrkv), [hiddens, hiddens*2], axis=-1)

    r = jax.nn.sigmoid(hk.Linear(hiddens * widening_factor, with_bias=False, w_init=initializer)(x_r))
    k = hk.Linear(hiddens * widening_factor, with_bias=False, w_init=initializer)(x_k)
    v = hk.Linear(hiddens * widening_factor, with_bias=False, w_init=initializer)(x_v)
    return r, k, v

class WKVLayer(hk.Module):
    def __init__(self,
                 init_scale: float,
                 widening_factor: int = 1,
                 name: Optional[str] = None,
                 layer_scale: float = 0,
                 ratio_0_to_1: float = 0,
                 batch_first: bool = True,
                 do_rearrange: bool = False):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor
        self._batch_first = batch_first
        self._rearrange = do_rearrange

        ratio_1_to_almost0 = 1.0 - layer_scale  # 1 to ~0
        def init3(shape, dtype):
            ddd = jnp.arange(shape[-1] // 3, dtype=dtype)
            ddd = ddd / (shape[-1] // 3)
            k = jnp.power(ddd, ratio_1_to_almost0)
            v = jnp.power(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1
            r = jnp.power(ddd, ratio_1_to_almost0 / 2)
            return jnp.concatenate([k, v, r]).reshape(shape)
        
        self._time_mix_init = init3

        def init_time_decay(shape, dtype):
            return jnp.array([-5 + 8 * (h / (shape[-1] - 1)) ** (0.7 + 1.3 * ratio_0_to_1) for h in range(shape[-1])], dtype=dtype)
        
        def init_time_first(shape, dtype):
            return jnp.ones(shape) * jnp.log(0.3) + (jnp.array([(i + 1) % 3 - 1 for i in range(shape[-1])], dtype=dtype) * 0.5)
        
        self._init_time_decay = init_time_decay
        self._init_time_first = init_time_first

    
    def __call__(self, x: jnp.ndarray, xx: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)

        # TODO: timeshift
        if not self._batch_first:
            # S B
            x_prev = jnp.concatenate([jnp.zeros_like(x[:1, ...]), x[:-1, ...]], axis=0)
        else:
            # B S
            x_prev = jnp.concatenate([jnp.zeros_like(x[:, :1, ...]), x[:, :-1, ...]], axis=1)
        r, k, v = rkv(x, x_prev, self._time_mix_init, self._init_scale, widening_factor=self._widening_factor)

        # w
        time_decay = hk.get_parameter("time_decay", (hiddens * self._widening_factor,), init=self._init_time_decay)
        # u
        time_first = hk.get_parameter("time_first", (hiddens * self._widening_factor,), init=self._init_time_first)

        if not self._batch_first or self._rearrange:
            if self._batch_first:
                # x is B S E
                k_ = rearrange(k, 'b s e -> s (b e)')
                v_ = rearrange(v, 'b s e -> s (b e)')
            else:
                # x is S B E
                k_ = k.reshape(k.shape[0], -1)
                v_ = v.reshape(v.shape[0], -1)
            # print(time_decay.shape, r.shape)
            W_ = repeat(time_decay, 'e -> s (b e)', s=r.shape[0], b=r.shape[1])
            expkv_, expk_, p_ = v_, jnp.ones_like(v_), k_
            a_state_, b_state_, _, p_state_ = jax.lax.associative_scan(assoc_reduce_step, (expkv_, expk_, W_, p_))
            a_state = rearrange(a_state_, 's (b e) -> s b e', b=r.shape[1])
            b_state = rearrange(b_state_, 's (b e) -> s b e', b=r.shape[1])
            p_state = rearrange(p_state_, 's (b e) -> s b e', b=r.shape[1])
            expkv   = rearrange(expkv_, 's (b e) -> s b e', b=r.shape[1])
            expk    = rearrange(expk_, 's (b e) -> s b e', b=r.shape[1])
            p       = rearrange(p_, 's (b e) -> s b e', b=r.shape[1])
            c, d, _ = exp_mix_frac(p_state, p + time_first + time_decay, a_state, b_state, expkv, expk)
            rwkv = c / d
        else:
            # x is B S E
            rwkv = WKV_gpu_batch(time_decay, time_first, k, v)
        return hk.Linear(hiddens, with_bias=False, w_init=initializer, name="output")(rwkv)

def create_layer(init_scale: float,
                 layer_id: int,
                 layers: int,
                 widening_factor: int = 1,
                 name: Optional[str] = None,
                 batch_first: bool = True,
                 do_rearrange: bool = False):
    return WKVLayer(init_scale, widening_factor, name, layer_scale=layer_id / layers, ratio_0_to_1=layer_id / (layers - 1), batch_first=batch_first, do_rearrange=do_rearrange)

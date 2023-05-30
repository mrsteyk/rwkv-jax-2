import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
# from einops import repeat

from typing import Optional, Callable

@jax.jit
def swiglu(x: jnp.ndarray) -> jnp.ndarray:
    h = x.shape[-1]
    i, g = jnp.split(x, [h//2], axis=-1)
    return jax.nn.silu(i) * g

@jax.jit
def relu_square(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.square(jax.nn.relu(x))

# Mish(x)=x∗Tanh(Softplus(x))
@jax.jit
def mish(x: jnp.ndarray) -> jnp.ndarray:
    return x * jnp.tanh(jax.nn.softplus(x))

class LinearMLP(hk.Module):
    def __init__(self,
                 init_scale: float,
                 widening_factor: int = 4,
                 name: Optional[str] = None,
                 activation_function: Callable = jax.nn.gelu,
                 # Mimics GPT-J
                 bias: bool = True,
                 skip_output = False):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor
        self._activation_function = activation_function
        self._with_bias = bias
        self._skip_output = skip_output
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        x = hk.Linear(self._widening_factor * hiddens, w_init=initializer, with_bias=self._with_bias)(x)
        x = self._activation_function(x)
        if not self._skip_output:
            return hk.Linear(hiddens, w_init=initializer)(x)
        else:
            return x

class LLaMAMLP(hk.Module):
    def __init__(self,
                 init_scale: float,
                 widening_factor: int = 4,
                 name: Optional[str] = None,
                 activation_function: Callable = jax.nn.silu,
                 # Mimics LLaMA
                 bias: bool = False,
                 skip_output = False):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor
        self._activation_function = activation_function
        self._with_bias = bias
        self._skip_output = skip_output
    
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        w1 = hk.Linear(self._widening_factor * hiddens, w_init=initializer, with_bias=self._with_bias, name="w1")(x)
        w3 = hk.Linear(self._widening_factor * hiddens, w_init=initializer, with_bias=self._with_bias, name="w3")(x)
        o = self._activation_function(w1) * w3
        if not self._skip_output:
            return hk.Linear(hiddens, w_init=initializer, with_bias=self._with_bias, name="w2")(o)
        else:
            return o

class ChannelMixing(hk.Module):
    def __init__(self,
                 init_scale: float,
                 widening_factor: int = 4,
                 name: Optional[str] = None,
                 activation_function: Callable = relu_square,
                 bias: bool = False,
                 # layer_id / n_layers
                 layer_scale: float = 0,
                 batch_first = True,
                 # Literally nothing!
                 skip_output = False):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor
        self._activation_function = activation_function
        self._with_bias = bias
        self._batch_first = batch_first

        def init(shape, dtype):
            ratio_1_to_almost0 = 1.0 - layer_scale  # 1 to ~0
            ddd = jnp.arange(shape[-1], dtype=dtype)
            ddd = ddd / shape[-1]
            # self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            # self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            return jnp.power(ddd, ratio_1_to_almost0).reshape(shape)

        # ecksde
        # def init2(shape):
        #     return jnp.concatenate([init(shape) for _ in range(2)], axis=-1)
        
        self._time_mix_init = init
    
    def __call__(self, x: jnp.ndarray, xx: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)
        # TODO: timeshift
        if xx is None:
            if not self._batch_first:
                # S B
                xx = jnp.concatenate([jnp.zeros_like(x[:1, ...]), x[:-1, ...]], axis=0)
            else:
                # B S
                xx = jnp.concatenate([jnp.zeros_like(x[:, :1, ...]), x[:, :-1, ...]], axis=1)
        
        tmk = hk.get_parameter("time_mix_k", (hiddens,), dtype=jnp.float32, init=self._time_mix_init)
        xk = x * tmk + xx * (1 - tmk)
        tmr = hk.get_parameter("time_mix_r", (hiddens,), dtype=jnp.float32, init=self._time_mix_init)
        xr = x * tmr + xx * (1 - tmr)

        k = hk.Linear(hiddens * self._widening_factor, with_bias=self._with_bias, w_init=initializer, name="key")(xk)
        k = self._activation_function(k)
        kv = hk.Linear(hiddens, with_bias=self._with_bias, w_init=initializer, name="value")(k)
        
        r = hk.Linear(hiddens, with_bias=self._with_bias, w_init=initializer, name="receptance")(xr)
        return jax.nn.sigmoid(r) * kv

class MishGLUMLP(hk.Module):
    def __init__(self,
                 init_scale: float,
                 widening_factor: int = 4,
                 name: Optional[str] = None,
                 activation_function: Callable = mish,
                 bias: bool = False,
                 # layer_id / n_layers
                 layer_scale: float = 0,
                 batch_first = True,
                 skip_output = False):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor
        self._activation_function = activation_function
        self._with_bias = bias
        self._batch_first = batch_first
        self._skip_output = skip_output

        def init(shape, dtype):
            ratio_1_to_almost0 = 1.0 - layer_scale  # 1 to ~0
            ddd = jnp.arange(shape[-1], dtype=dtype)
            ddd = ddd / shape[-1]
            # self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            # self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            return jnp.power(ddd, ratio_1_to_almost0).reshape(shape)

        # ecksde
        def init2(shape, dtype):
            new_shape = [*shape[:-1], shape[-1] // 2]
            return jnp.concatenate([init(new_shape, dtype) for _ in range(2)], axis=-1)
        
        self._time_mix_init = init2
    
    def __call__(self, x: jnp.ndarray, xx: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)

        # TODO: timeshift
        if xx is None:
            if not self._batch_first:
                # S B
                xx = jnp.concatenate([jnp.zeros_like(x[:1, ...]), x[:-1, ...]], axis=0)
            else:
                # B S
                xx = jnp.concatenate([jnp.zeros_like(x[:, :1, ...]), x[:, :-1, ...]], axis=1)
        
        xx = jnp.concatenate([xx, xx], axis=-1)
        x = jnp.concatenate([x, x], axis=-1)
        tmkr = hk.get_parameter("time_mix_kr", (hiddens * 2,), dtype=jnp.float32, init=self._time_mix_init)
        xab = x * tmkr + xx * (1 - tmkr)

        ab = hk.Linear(hiddens * self._widening_factor * 2, with_bias=self._with_bias, w_init=initializer)(xab)
        a, b = jnp.split(ab, [hiddens * self._widening_factor], axis=-1)

        o = a * self._activation_function(b)
        if not self._skip_output:
            return hk.Linear(hiddens, with_bias=self._with_bias, w_init=initializer)(o)
        else:
            return o

class LinearJ2MLP(hk.Module):
    def __init__(self,
                 init_scale: float,
                 widening_factor: int = 4,
                 name: Optional[str] = None,
                 # You can swap this for swiglu idfk
                 activation_function: Callable = jax.nn.glu,
                 # Mimics GPT-J2
                 bias: bool = True,
                 # no op
                 skip_output: bool = True):
        super().__init__(name=name)
        self._init_scale = init_scale
        self._widening_factor = widening_factor
        self._activation_function = activation_function
        self._with_bias = bias
        # This was originally a dirty fix!
        assert skip_output

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        hiddens = x.shape[-1]
        initializer = hk.initializers.VarianceScaling(self._init_scale)

        # TODO: original implementation doesn't set w_init
        # Under default conditions it will be hiddens * 4 * 2 = hiddens * 8, as per original!
        x = hk.Linear(hiddens * self._widening_factor * 2, with_bias=self._with_bias, w_init=initializer)(x)

        return self._activation_function(x)

TRANSFORMER_LAYER_MAPPING = {
    "llama": (LLaMAMLP, jax.nn.silu, False),
    "linear": (LinearMLP, jax.nn.gelu, True),
    
    # !!!ONLY FOR USE IN J2 RESIDUAL!!!
    "linearj2": (LinearJ2MLP, jax.nn.glu, True),
    # Explicit no bias, has no effect with current default char level config!
    "linearj2_nb": (LinearJ2MLP, jax.nn.glu, False),
    # PaLM didn't use bias in dense(MLP) layers
    "linearj2_swiglu": (LinearJ2MLP, swiglu, False),
}

TRANSFORMER_LAYERS = TRANSFORMER_LAYER_MAPPING.keys()

def create_transformer_layer(layer_name: str, init_scale: float, widening_factor: float = 4, name: Optional[str] = None, activation_function: Optional[Callable] = None, bias: Optional[bool] = None, skip_output = False):
    assert layer_name in TRANSFORMER_LAYERS
    # I think that's how python works?
    func, act_def, bias_def = TRANSFORMER_LAYER_MAPPING[layer_name]
    if activation_function is None:
        activation_function = act_def
    if bias is None:
        bias = bias_def
    return func(init_scale, widening_factor, name, activation_function, bias, skip_output=skip_output)

RWKV_LAYER_MAPPINGS = {
    "rwkv": (ChannelMixing, relu_square, False),
    "mishglu": (MishGLUMLP, mish, False),
}
RWKV_LAYERS = RWKV_LAYER_MAPPINGS.keys()

def create_rwkv_layer(layer_name: str, init_scale: float, layer_id: int, layers: int, widening_factor: float = 4, name: Optional[str] = None, activation_function: Optional[Callable] = None, bias: Optional[bool] = None, batch_first = True, skip_output = False):
    assert layer_name in RWKV_LAYERS
    # I think that's how python works?
    func, act_def, bias_def = RWKV_LAYER_MAPPINGS[layer_name]
    if activation_function is None:
        activation_function = act_def
    if bias is None:
        bias = bias_def
    return func(init_scale, widening_factor, name, activation_function, bias, layer_scale=layer_id / layers, batch_first=batch_first, skip_output=skip_output)

if __name__ == "__main__":
    def test(x):
        a = create_transformer_layer("llama", 1)
        return a(x)
    
    f = hk.transform(test)
    r = jax.random.PRNGKey(0)
    state = f.init(r, jnp.zeros((1, 1, 512), dtype=jnp.float32))
    print(state.keys())
    f.apply(state, None, jnp.zeros((1, 1, 512), dtype=jnp.float32))
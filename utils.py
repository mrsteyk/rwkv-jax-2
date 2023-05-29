import jax
import optax
import numpy as np
import jax.numpy as jnp

from functools import partial

@partial(jax.jit, static_argnums=3)
def opt_jit(grad_acc, opt_state, params, optimizer):
    total_grad = jax.tree_map(lambda x: jnp.mean(x, axis=0), grad_acc)

    cpu_device = jax.devices("cpu")[0]

    total_grad = jax.device_put(total_grad, device=cpu_device)
    cpu_params = jax.device_put(jax.tree_map(lambda x: x[0], params), device=cpu_device)

    updates, new_opt_state = optimizer.update(total_grad, opt_state)

    new_params = optax.apply_updates(cpu_params, updates)

    new_grad_acc = jax.tree_map(jnp.zeros_like, grad_acc)
    return new_grad_acc, new_opt_state, new_params


def opt_state(state, optimizer):
    new_grad_acc, new_opt_state, new_params = opt_jit(state["grad_acc"],
                                                      state["opt_state"],
                                                      state["params"],
                                                      optimizer)

    state["grad_acc"] = new_grad_acc
    state["opt_state"] = new_opt_state
    state["params"] = jax.device_put_replicated(new_params, jax.local_devices())
    state["grad_count"] = np.array(0)
    return state

def init_fn(master_rng, data, init_fn, optimizer):
    out_rng, init_rng = jax.random.split(master_rng)

    # copy the same initial params to each accelerator
    init_rng = jnp.broadcast_to(init_rng, (jax.local_device_count(),) + init_rng.shape)
    params = jax.pmap(init_fn)(init_rng, data)

    cpu_device = jax.devices("cpu")[0]

    # place optimizer state on CPU
    cpu_params = jax.tree_map(lambda x: jax.device_put(x[0], device=cpu_device), params)
    opt_state = optimizer.init(cpu_params)

    return dict(
        step=np.array(0),
        rng=out_rng,
        opt_state=opt_state,
        grad_acc=jax.tree_map(jnp.zeros_like, params),
        grad_count=np.array(0),
        params=params)
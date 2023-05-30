import jax
import optax
import operator
import functools
import time
import haiku as hk

import model
import utils

if __name__ == "__main__":
    BB = jax.local_device_count()
    B = 2
    T = 2**11
    rwkv, config = model.RWKV_CharLevel(mlp="mishglu", j_residual=False, j2_residual=True, j3_residual=False)
    # C = config["n_embd"]
    optimizer = optax.lion(1e-5)
    state = utils.init_fn(jax.random.PRNGKey(42), jax.numpy.ones((BB, T, B), dtype=int), rwkv.init, optimizer)
    # apply_fn = jax.jit(rwkv.apply)

    print(hk.data_structures.tree_size(state['params'])/1000_000)

    # print(state)

    fn_grad = jax.jit(model.loss_fn_grad(rwkv.apply))
    @functools.partial(jax.pmap, donate_argnums=3)
    def step_fn(x, y, params, acc):
        loss, grad = fn_grad(x, y, params)

        cpu_device = jax.devices("cpu")[0]
        cpu_params = jax.tree_map(lambda x: jax.device_put(x[0], device=cpu_device), grad)
        del grad
        # print(acc, grad)
        acc = jax.tree_map(operator.add, acc, cpu_params)
        # grad = None
        return loss, acc

    start = time.time()
    loss, acc = step_fn(jax.numpy.ones((BB, T, B), dtype=int), jax.numpy.ones((BB, T, B), dtype=int), state['params'], state['grad_acc'])
    # acc = jax.tree_map(lambda x: x.astype(jax.numpy.bfloat16), acc)
    print("A", time.time() - start)
    start = time.time()
    loss, acc = step_fn(jax.numpy.ones((BB, T, B), dtype=int), jax.numpy.ones((BB, T, B), dtype=int), state['params'], acc)
    dur = time.time() - start
    print("B", dur, 1/dur, (1/dur) * BB * B * T)
    loss = loss.mean()
    print(loss)
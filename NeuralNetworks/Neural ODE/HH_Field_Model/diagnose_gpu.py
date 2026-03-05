"""
GPU Compilation Diagnostic

Runs increasingly complex JIT compilations to find exactly
where the XLA GPU compiler chokes. Run this on your WSL GPU machine:

    python diagnose_gpu.py

Each test has a 60-second timeout. The first test that hangs is the culprit.
"""

import os
import sys
import time
import signal

# Try these XLA flags to speed up GPU compilation
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_latency_hiding_scheduler=false '
    '--xla_gpu_graph_level=0'
)
# Persistent compilation cache — reuse across runs
os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'
os.environ['JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS'] = '0'

import jax
import jax.numpy as jnp
import equinox as eqx
import optax


def timed_test(name, fn, timeout=120):
    """Run fn() with a timeout. Returns True if it completed."""
    print(f"\n{'='*60}")
    print(f"TEST: {name}")
    print(f"{'='*60}")
    sys.stdout.flush()

    t0 = time.time()
    try:
        result = fn()
        jax.block_until_ready(result)
        elapsed = time.time() - t0
        print(f"  PASS  ({elapsed:.2f}s)")
        sys.stdout.flush()
        return True
    except Exception as e:
        elapsed = time.time() - t0
        print(f"  FAIL  ({elapsed:.2f}s): {e}")
        sys.stdout.flush()
        return False


def main():
    print("GPU Compilation Diagnostic")
    print(f"JAX version: {jax.__version__}")
    print(f"Backend: {jax.default_backend()}")
    print(f"Devices: {jax.devices()}")

    if jax.default_backend() != 'gpu':
        print("\nWARNING: Not running on GPU! This diagnostic is for GPU issues.")
        print("Make sure JAX GPU is installed: pip install jax[cuda12]")

    key = jax.random.PRNGKey(42)

    # ================================================================
    # Test 1: Bare minimum — can JAX JIT anything on GPU?
    # ================================================================
    def test1():
        @jax.jit
        def f(x):
            return x * 2.0 + 1.0
        return f(jnp.ones(100))

    if not timed_test("1. Basic JIT (scalar ops)", test1):
        print("\n*** JAX GPU is broken at the most basic level! ***")
        print("Check: CUDA version, GPU driver, jax[cuda] installation")
        return

    # ================================================================
    # Test 2: Simple matmul
    # ================================================================
    def test2():
        W = jax.random.normal(key, (256, 128))
        b = jnp.zeros(256)
        x = jax.random.normal(key, (8192, 128))

        @jax.jit
        def f(x, W, b):
            return jnp.tanh(x @ W.T + b)
        return f(x, W, b)

    if not timed_test("2. Single matmul (8192, 128) @ (128, 256)", test2):
        print("\n*** Basic matmul fails — this is a GPU driver issue ***")
        return

    # ================================================================
    # Test 3: Chained matmuls (what the MLP does, unrolled)
    # ================================================================
    def test3():
        W1 = jax.random.normal(key, (256, 128))
        b1 = jnp.zeros(256)
        W2 = jax.random.normal(key, (256, 256))
        b2 = jnp.zeros(256)
        W3 = jax.random.normal(key, (4, 256))
        b3 = jnp.zeros(4)
        x = jax.random.normal(key, (8192, 128))

        @jax.jit
        def f(x):
            x = jnp.tanh(x @ W1.T + b1)
            x = jnp.tanh(x @ W2.T + b2)
            return x @ W3.T + b3
        return f(x)

    if not timed_test("3. Chained matmuls (3 layers, unrolled)", test3):
        print("\n*** Chained matmul fails — unlikely, but check GPU memory ***")
        return

    # ================================================================
    # Test 4: lax.scan matmul (what the new model does)
    # ================================================================
    def test4():
        W_in = jax.random.normal(key, (256, 128))
        b_in = jnp.zeros(256)
        W_hidden = jax.random.normal(key, (3, 256, 256))
        b_hidden = jnp.zeros((3, 256))
        W_out = jax.random.normal(key, (4, 256))
        b_out = jnp.zeros(4)
        x = jax.random.normal(key, (8192, 128))

        @jax.jit
        def f(x):
            x = jnp.tanh(x @ W_in.T + b_in)
            def body(x, wb):
                w, b = wb
                return jnp.tanh(x @ w.T + b), None
            x, _ = jax.lax.scan(body, x, (W_hidden, b_hidden))
            return x @ W_out.T + b_out
        return f(x)

    if not timed_test("4. lax.scan matmul (3 hidden layers via scan)", test4):
        print("\n*** lax.scan matmul fails — try unrolled approach instead ***")
        return

    # ================================================================
    # Test 5: Gradient of matmul chain (the backward pass)
    # ================================================================
    def test5():
        W1 = jax.random.normal(key, (256, 128))
        b1 = jnp.zeros(256)
        W2 = jax.random.normal(key, (256, 256))
        b2 = jnp.zeros(256)
        W3 = jax.random.normal(key, (4, 256))
        b3 = jnp.zeros(4)
        x = jax.random.normal(key, (8192, 128))
        targets = jax.random.normal(key, (8192, 4))

        @jax.jit
        def loss_and_grad(W1, b1, W2, b2, W3, b3):
            h = jnp.tanh(x @ W1.T + b1)
            h = jnp.tanh(h @ W2.T + b2)
            pred = h @ W3.T + b3
            return jnp.mean((pred - targets) ** 2)

        return jax.jit(jax.grad(loss_and_grad, argnums=(0,1,2,3,4,5)))(W1, b1, W2, b2, W3, b3)

    if not timed_test("5. Gradient of chained matmuls", test5):
        print("\n*** Backward pass hangs — this is the gradient compilation ***")
        return

    # ================================================================
    # Test 6: Gradient of lax.scan matmul
    # ================================================================
    def test6():
        W_in = jax.random.normal(key, (256, 128))
        b_in = jnp.zeros(256)
        W_hidden = jax.random.normal(key, (3, 256, 256))
        b_hidden = jnp.zeros((3, 256))
        W_out = jax.random.normal(key, (4, 256))
        b_out = jnp.zeros(4)
        x = jax.random.normal(key, (8192, 128))
        targets = jax.random.normal(key, (8192, 4))

        @jax.jit
        def loss_and_grad(W_in, b_in, W_hidden, b_hidden, W_out, b_out):
            h = jnp.tanh(x @ W_in.T + b_in)
            def body(h, wb):
                w, b = wb
                return jnp.tanh(h @ w.T + b), None
            h, _ = jax.lax.scan(body, h, (W_hidden, b_hidden))
            pred = h @ W_out.T + b_out
            return jnp.mean((pred - targets) ** 2)

        return jax.jit(jax.grad(loss_and_grad, argnums=(0,1,2,3,4,5)))(
            W_in, b_in, W_hidden, b_hidden, W_out, b_out)

    if not timed_test("6. Gradient of lax.scan matmul", test6):
        print("\n*** Backward pass through lax.scan hangs ***")
        print("*** Root cause: XLA GPU can't handle scan gradients efficiently ***")
        print("*** Solution: Use unrolled Python for loop (only 3-5 layers) ***")
        return

    # ================================================================
    # Test 7: Equinox model forward + backward (the actual training step)
    # ================================================================
    def test7():
        from model import create_model
        from losses import field_loss

        model = create_model(key=key)
        x = jax.random.normal(key, (8192, 4))
        I = jax.random.normal(key, (8192,))
        targets = jax.random.normal(key, (8192, 4))

        @eqx.filter_jit
        def step(model):
            (loss, info), grads = eqx.filter_value_and_grad(field_loss, has_aux=True)(
                model, x, I, targets
            )
            return loss, grads

        return step(model)

    if not timed_test("7. Equinox model forward + backward (field_loss)", test7):
        print("\n*** Equinox filter_value_and_grad hangs ***")
        print("*** Check if test 5 or 6 also failed ***")
        return

    # ================================================================
    # Test 8: Full training step with optimizer
    # ================================================================
    def test8():
        from model import create_model
        from losses import field_loss

        model = create_model(key=key)
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(1e-3, weight_decay=1e-4),
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        x = jax.random.normal(key, (8192, 4))
        I = jax.random.normal(key, (8192,))
        targets = jax.random.normal(key, (8192, 4))

        @eqx.filter_jit
        def step(model, opt_state):
            (loss, info), grads = eqx.filter_value_and_grad(field_loss, has_aux=True)(
                model, x, I, targets
            )
            grads = jax.tree.map(
                lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads
            )
            updates, opt_state_new = optimizer.update(grads, opt_state, model)
            model_new = eqx.apply_updates(model, updates)
            return model_new, opt_state_new, loss

        return step(model, opt_state)

    if not timed_test("8. Full training step with optimizer", test8):
        print("\n*** Optimizer chain or NaN cleaning causes the hang ***")
        print("*** Try removing clip_by_global_norm or NaN filter ***")
        return

    # ================================================================
    # Test 9: The cosine LR schedule (what train_field.py actually uses)
    # ================================================================
    def test9():
        from model import create_model
        from losses import field_loss

        model = create_model(key=key)
        lr_schedule = optax.cosine_decay_schedule(
            init_value=1e-3, decay_steps=5000, alpha=0.01
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(lr_schedule, weight_decay=1e-4),
        )
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        x = jax.random.normal(key, (8192, 4))
        I = jax.random.normal(key, (8192,))
        targets = jax.random.normal(key, (8192, 4))

        @eqx.filter_jit
        def step(model, opt_state):
            (loss, info), grads = eqx.filter_value_and_grad(field_loss, has_aux=True)(
                model, x, I, targets
            )
            grads = jax.tree.map(
                lambda g: jnp.where(jnp.isfinite(g), g, 0.0), grads
            )
            updates, opt_state_new = optimizer.update(grads, opt_state, model)
            model_new = eqx.apply_updates(model, updates)
            return model_new, opt_state_new, loss

        return step(model, opt_state)

    if not timed_test("9. Full step with cosine LR schedule", test9):
        print("\n*** Cosine schedule causes the hang ***")
        print("*** Try constant LR instead ***")
        return

    print("\n" + "="*60)
    print("ALL TESTS PASSED!")
    print("="*60)
    print("\nAll 9 tests compiled on GPU. If train_field.py still hangs,")
    print("the issue might be in the data loading/sampling pipeline,")
    print("or in a lazy compilation triggered later in training.")
    print("\nTry running train_field.py with this at the top:")
    print("  os.environ['XLA_FLAGS'] = '--xla_gpu_enable_latency_hiding_scheduler=false --xla_gpu_graph_level=0'")
    print("  os.environ['JAX_COMPILATION_CACHE_DIR'] = '/tmp/jax_cache'")


if __name__ == "__main__":
    main()

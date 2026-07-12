"""
Mini Distributed Training and Memory-Constrained Trainer from Scratch in NumPy

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - make_synthetic_regression_batch
import numpy as np

def make_synthetic_regression_batch(batch_size, in_dim, out_dim, seed):
    """Return (x, y) where x is (batch_size, in_dim) and y is (batch_size, out_dim) float64."""
    # TODO: seed numpy, sample x, build a hidden teacher, and produce noisy targets y.
    np.random.seed(seed)

    x = np.random.randn(batch_size, in_dim)
    hidden = np.random.randn(in_dim, out_dim)
    noise = np.random.normal(loc=0, scale=0.1, size=(batch_size, out_dim))

    y = x @ hidden + noise

    return x, y

# Step 2 - init_mlp_params
import numpy as np

def init_mlp_params(in_dim, hidden_dim, out_dim, seed):
    # TODO: return a dict {'W1','b1','W2','b2'} with He-initialized weights and zero biases.
    np.random.seed(seed)

    W1 = np.random.normal(loc=0, scale=np.sqrt(2/in_dim), size = (in_dim, hidden_dim))
    W2 = np.random.normal(loc=0, scale=np.sqrt(2/hidden_dim), size = (hidden_dim, out_dim))

    b1 = np.zeros((hidden_dim, ))
    b2 = np.zeros((out_dim, ))

    return {
        'W1': W1,
        'W2': W2,
        'b1': b1,
        'b2': b2
    }

# Step 3 - linear_forward
def linear_forward(x, w, b):
    # TODO: apply y = x @ w + b and return the resulting (N, out_dim) array
    return x @ w + b.reshape(1, -1)

# Step 4 - relu_forward
def relu_forward(x):
    # TODO: apply the ReLU activation elementwise and return an array of the same shape.
    
    mask = x > 0
    x_masked = np.where(mask, x, 0)

    return x_masked

# Step 5 - mlp_forward
def mlp_forward(x, params):
    # TODO: run the two-layer MLP forward and return (y_pred, cache) with keys 'x','z1','a1','z2'.
    z1 = linear_forward(x, params['W1'], params['b1'])
    a1 = relu_forward(z1)

    z2 = linear_forward(a1, params['W2'], params['b2'])

    cache = {}
    cache['x'] = x
    cache['z1'] = z1
    cache['a1'] = a1
    cache['z2'] = z2

    return z2, cache

# Step 6 - mse_loss_and_grad
def mse_loss_and_grad(y_pred, y_true):
    # TODO: compute mean squared error loss and its gradient with respect to y_pred
    diff = y_pred - y_true
    n = y_pred.size

    loss = np.sum(diff ** 2) / n
    grad = 2 * diff / n

    return loss, grad

# Step 7 - linear_backward
import numpy as np

def linear_backward(d_out, x, w):
    # TODO: backprop through y = x @ w + b and return (dx, dw, db)
    # d_out: N, out
    dx = d_out @ w.T
    dw = x.T @ d_out
    db = np.sum(d_out, axis = 0)

    return dx, dw, db

# Step 8 - relu_backward
def relu_backward(d_out, z):
    # TODO: backprop through ReLU using the pre-activation z, return dz with same shape.
    
    return d_out * (z>0)

# Step 9 - first_linear_backward
def first_linear_backward(d_z1, x, w1):
    # TODO: return gradients (dx, dW1, db1) for z1 = x @ w1 + b1 given d_z1.
    return linear_backward(d_z1, x, w1)

# Step 10 - mlp_backward
def mlp_backward(dy_pred, cache, params):
    # TODO: run the full MLP backward pass returning grads dict with keys W1,b1,W2,b2
    
    # z2 = a1 @ w2 + b2
    da1, dW2, db2 = linear_backward(dy_pred, cache['a1'], params['W2'])

    # a1 = ReLU(z1)
    dz1 = relu_backward(da1, cache['z1'])

    # z1 = x @ W1 + b1
    dx, dW1, db1 = linear_backward(dz1, cache['x'], params['W1'])

    return {
        'W1': dW1,
        'b1': db1,
        'W2': dW2,
        'b2': db2
    }

# Step 11 - split_into_micro_batches
def split_into_micro_batches(x, y, micro_batch_size):
    # TODO: split (x, y) into contiguous micro batches of at most micro_batch_size rows.
    if micro_batch_size <= 0:
        raise ValueError("micro_batch_size must be positive")

    if len(x) != len(y):
        raise ValueError("x and y must have the same number of rows")

    return [
        (
            x[start:start + micro_batch_size],
            y[start:start + micro_batch_size],
        )
        for start in range(0, len(x), micro_batch_size)
    ]

# Step 12 - accumulate_gradients
def accumulate_gradients(accum_grads, new_grads):
    # TODO: return a dict whose values are elementwise sums of accum_grads and new_grads.
    if accum_grads is None:
        return {key: value.copy()
                for key, value in new_grads.items()
                }

    return {key: accum_grads[key] + new_grads[key] for key in new_grads}

# Step 13 - scale_accumulated_gradients
def scale_accumulated_gradients(accum_grads, num_micro_batches):
    # TODO: divide each gradient tensor by num_micro_batches and return a new dict
    return {
        key: gradient / num_micro_batches
        for key, gradient in accum_grads.items()
    }

# Step 14 - grad_accumulation_step
def grad_accumulation_step(x, y, params, micro_batch_size):
    # TODO: run forward/backward on each micro batch and combine grads to match a full-batch step.
    
    micro_batches = split_into_micro_batches(x, y, micro_batch_size)
    accum_grads = None

    for x_micro, y_micro in micro_batches:
        y_pred, cache = mlp_forward(x_micro, params)
        _, dy_pred = mse_loss_and_grad(y_pred, y_micro)
        new_grads = mlp_backward(dy_pred, cache, params)

        accum_grads = accumulate_gradients(accum_grads, new_grads)

    return scale_accumulated_gradients(
        accum_grads,
        len(micro_batches)
    )

# Step 15 - mlp_forward_checkpointed
def mlp_forward_checkpointed(x, params):
    # TODO: forward pass that caches only the block input x, not intermediates.
    z1 = linear_forward(x, params['W1'], params['b1'])
    a1 = relu_forward(z1)

    z2 = linear_forward(a1, params['W2'], params['b2'])

    cache = {}
    cache['x'] = x

    return z2, cache

# Step 16 - recompute_block_activations
def recompute_block_activations(x, params):
    # TODO: recompute z1, a1, z2 from x and params and return them in a cache dict
    
    z1 = linear_forward(x, params['W1'], params['b1'])
    a1 = relu_forward(z1)
    z2 = linear_forward(a1, params['W2'], params['b2'])

    cache = {}
    cache['x'] = x
    cache['z1'] = z1
    cache['a1'] = a1
    cache['z2'] = z2

    return cache

# Step 17 - mlp_backward_checkpointed
def mlp_backward_checkpointed(dy_pred, light_cache, params):
    # TODO: recompute activations from light_cache['x'] and run the standard MLP backward
    cache = recompute_block_activations(light_cache['x'], params)

    # z2 = a1 @ W2 + b2
    da1, dW2, db2 = linear_backward(dy_pred, cache['a1'], params['W2'])

    # a1 = relu(z1)
    dz1 = relu_backward(da1, cache['z1'])

    # z1 = x @ W1 + b1
    dx, dW1, db1 = linear_backward(dz1, cache['x'], params['W1'])

    return {
        'W1': dW1,
        'b1': db1,
        'W2': dW2,
        'b2': db2
    }

# Step 18 - estimate_checkpointing_memory_savings
def estimate_checkpointing_memory_savings(batch_size, in_dim, hidden_dim, out_dim, dtype_bytes):
    # TODO: estimate activation memory in bytes for full vs checkpointed forward on the two-layer MLP.
    # x:  (batch_size, in_dim)
    # z1: (batch_size, hidden_dim)
    # a1: (batch_size, hidden_dim)

    full_elements = batch_size * (
        in_dim + 2 * hidden_dim
        )

    checkpointed_elements = batch_size * in_dim

    full_bytes = full_elements * dtype_bytes
    checkpoint_bytes = checkpointed_elements * dtype_bytes
    saved_bytes = full_bytes - checkpoint_bytes

    return {
        "full_bytes": full_bytes,
        "checkpoint_bytes": checkpoint_bytes,
        "saved_bytes": saved_bytes,
    }

# Step 19 - cast_to_half_precision
def cast_to_half_precision(values):
    # TODO: Return a new dict mapping each key to its array converted to float16.
    out = {}
    for val in values.keys():
        out[val] = values[val].astype(np.float16, copy=True)

    return out

# Step 20 - make_master_params
def make_master_params(params):
    # TODO: return a dict mapping the same keys to independent float32 copies of each array.
    
    out = {}
    for key in params:
        out[key] = params[key].astype(np.float32, copy=True)

    return out

# Step 21 - scale_loss
def scale_loss(loss, dy_pred, scale):
    # TODO: Scale the scalar loss and the upstream gradient dy_pred by the fixed loss scale.
    return loss * scale, dy_pred * scale

# Step 22 - unscale_gradients (not yet solved)
# TODO: implement

# Step 23 - has_non_finite_gradients (not yet solved)
# TODO: implement

# Step 24 - mixed_precision_step (not yet solved)
# TODO: implement

# Step 25 - shard_dataset_across_workers (not yet solved)
# TODO: implement

# Step 26 - compute_local_gradients (not yet solved)
# TODO: implement

# Step 27 - all_reduce_mean (not yet solved)
# TODO: implement

# Step 28 - ring_all_reduce_mean (not yet solved)
# TODO: implement

# Step 29 - data_parallel_train_step (not yet solved)
# TODO: implement

# Step 30 - bucket_gradients (not yet solved)
# TODO: implement

# Step 31 - init_adam_state (not yet solved)
# TODO: implement

# Step 32 - partition_optimizer_state (not yet solved)
# TODO: implement

# Step 33 - local_shard_adam_update (not yet solved)
# TODO: implement

# Step 34 - all_gather_param_shards (not yet solved)
# TODO: implement

# Step 35 - zero_optimizer_step (not yet solved)
# TODO: implement

# Step 36 - compute_param_memory_bytes (not yet solved)
# TODO: implement

# Step 37 - compute_optimizer_memory_bytes (not yet solved)
# TODO: implement

# Step 38 - compute_peak_activation_memory_bytes (not yet solved)
# TODO: implement

# Step 39 - compare_memory_with_and_without_optimizations (not yet solved)
# TODO: implement

# Step 40 - full_distributed_training_loop (not yet solved)
# TODO: implement


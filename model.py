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

# Step 22 - unscale_gradients
def unscale_gradients(grads, scale):
    # TODO: divide every gradient tensor by scale and return a new float32 dict
    out = {}
    for key in grads.keys():
        out[key] = grads[key]/scale
        out[key] = out[key].astype(np.float32)

    return out

# Step 23 - has_non_finite_gradients
def has_non_finite_gradients(grads):
    # TODO: return True if any array in grads contains NaN or Inf, else False
    for nm, weight in grads.items():
        if weight is None:
            return True
        
        if not np.all(np.isfinite(weight)):
            return True
    
    return False

# Step 24 - mixed_precision_step
def mixed_precision_step(x, y, master_params, scale, lr):
    # TODO: run fp16 forward/backward, unscale grads, skip on overflow, else SGD-update fp32 master.
    # FP16 working copies
    params_half = cast_to_half_precision(master_params)
    x_half = x.astype(np.float16)
    y_half = y.astype(np.float16)

    # FP16 forward and backward
    y_pred, cache = mlp_forward(x_half, params_half)
    loss, dy_pred = mse_loss_and_grad(y_pred, y_half)

    _, scaled_dy_pred = scale_loss(loss, dy_pred, scale)
    scaled_grads = mlp_backward(
                                scaled_dy_pred,
                                cache,
                                params_half
                                )

    # Detect overflow before updating.
    overflow = has_non_finite_gradients(scaled_grads)

    # Always return an independent FP32 master copy.
    new_master = {
                name: param.astype(np.float32, copy=True)
                for name, param in master_params.items()
                }

    if overflow:
        return loss, new_master, True

    grads = unscale_gradients(scaled_grads, scale)

    # Defensive check after unscaling
    for name in new_master:
        new_master[name] -= lr * grads[name]

    return loss, new_master, False

# Step 25 - shard_dataset_across_workers
def shard_dataset_across_workers(x, y, num_workers):
    # TODO: split x and y into num_workers contiguous shards along axis 0
    
    x_shards = np.array_split(x, num_workers, axis=0)
    y_shards = np.array_split(y, num_workers, axis=0)

    return list(zip(x_shards, y_shards))

# Step 26 - compute_local_gradients
def compute_local_gradients(x, y, params):
    """Compute parameter gradients for one worker's data shard.

    Forward (mlp_forward) -> loss gradient (mse_loss_and_grad) -> backward
    (mlp_backward). Return a grads dict with keys 'W1', 'b1', 'W2', 'b2'.
    """
    # TODO: forward, then mse loss gradient, then backward; return grads
    
    y_pred, cache = mlp_forward(x, params)
    loss, dy_pred = mse_loss_and_grad(y_pred, y)
    grads = mlp_backward(dy_pred, cache, params)

    return grads

# Step 27 - all_reduce_mean
def all_reduce_mean(per_worker_grads):
    # TODO: average a list of gradient dicts elementwise across workers
    
    params = per_worker_grads[0].keys()
    out = {}

    for param in params:
        out[param] = np.mean([grads[param] for grads in per_worker_grads], axis = 0)

    return out

# Step 28 - ring_all_reduce_mean
def ring_all_reduce_mean(per_worker_arrays):
    # TODO: average arrays across workers via ring reduce-scatter then all-gather over chunks.

    arrays = [np.asarray(arr) for arr in per_worker_arrays]
    num_workers = len(arrays)
    original_shape = arrays[0].shape

    # Use floating point so integer inputs can produce a fractional mean.
    dtype = np.result_type(
        *(arr.dtype for arr in arrays),
        np.float64
    )

    num_elements = arrays[0].size

    # Pad so the flattened arrays can be split into equal-sized chunks.
    padded_size = (
        ((num_elements + num_workers - 1) // num_workers) * num_workers
        if num_elements > 0
        else 0
    )

    worker_chunks = []

    for arr in arrays:
        flat = arr.astype(dtype, copy=False).reshape(-1)

        if padded_size > num_elements:
            flat = np.pad(
                flat,
                (0, padded_size - num_elements),
                mode="constant"
            )

        chunks = [
            chunk.copy()
            for chunk in np.split(flat, num_workers)
        ]
        worker_chunks.append(chunks)

    # ------------------------------------------------------------
    # Phase 1: ring reduce-scatter
    # ------------------------------------------------------------
    for step in range(num_workers - 1):
        sends = []

        # All workers send simultaneously.
        for rank in range(num_workers):
            send_chunk_index = (rank - step) % num_workers
            sends.append(
                worker_chunks[rank][send_chunk_index].copy()
            )

        # Each worker receives from its previous neighbor and adds.
        for rank in range(num_workers):
            previous_rank = (rank - 1) % num_workers
            receive_chunk_index = (
                rank - step - 1
            ) % num_workers

            worker_chunks[rank][receive_chunk_index] += (
                sends[previous_rank]
            )

    # After reduce-scatter, worker rank owns fully reduced chunk:
    #
    #     (rank + 1) % num_workers

    # ------------------------------------------------------------
    # Phase 2: ring all-gather
    # ------------------------------------------------------------
    for step in range(num_workers - 1):
        sends = []

        # Send the reduced chunk owned or received in the prior step.
        for rank in range(num_workers):
            send_chunk_index = (
                rank - step + 1
            ) % num_workers

            sends.append(
                worker_chunks[rank][send_chunk_index].copy()
            )

        # Receive a completed chunk from the previous worker.
        for rank in range(num_workers):
            previous_rank = (rank - 1) % num_workers
            receive_chunk_index = (
                rank - step
            ) % num_workers

            worker_chunks[rank][receive_chunk_index] = (
                sends[previous_rank]
            )

    # Every worker now holds the complete globally summed tensor.
    summed_flat = np.concatenate(worker_chunks[0])

    # Remove padding and convert the sum into a mean.
    mean_flat = summed_flat[:num_elements] / num_workers

    return mean_flat.reshape(original_shape)

# Step 29 - data_parallel_train_step
def data_parallel_train_step(x, y, params, num_workers, lr):
    # TODO: shard the batch, compute local gradients, all-reduce mean them, then SGD update params.
    shards = shard_dataset_across_workers(x, y, num_workers)

    per_worker_grads = [compute_local_gradients(x_shard, y_shard, params) for x_shard, y_shard in shards]

    grads = all_reduce_mean(per_worker_grads)

    new_params = {}
    for key in params.keys():
        new_params[key] = params[key] - lr * grads[key]

    return new_params

# Step 30 - bucket_gradients
def bucket_gradients(grads, bucket_size):
    # TODO: pack flattened gradients into fixed-size buckets and return (buckets, meta).
    buckets = []
    meta = []

    current_parts = []
    current_size = 0
    current_bucket_index = 0

    for name in sorted(grads):
        grad = np.asarray(grads[name])
        flat_grad = np.ravel(grad)
        grad_size = flat_grad.size

        if current_parts and current_size + grad_size > bucket_size:
            buckets.append(np.concatenate(current_parts))

            current_parts = []
            current_size = 0
            current_bucket_index += 1
        
        start = current_size
        end = start + grad_size

        current_parts.append(flat_grad)
        meta.append(
            (
                name,
                grad.shape,
                start,
                end,
                current_bucket_index,
            )
        )

        current_size = end

    # Finalize the last bucket
    if current_parts:
        buckets.append(np.concatenate(current_parts))

    return buckets, meta

# Step 31 - init_adam_state
def init_adam_state(params):
    # TODO: build Adam state with zero first/second moments per param and step counter t=0.
    
    state = {}

    opt_params = ['m', 'v']

    for opt_param in opt_params:
        state[opt_param] = {}
        for name in params.keys():
            shape = params[name].shape
            dtype = params[name].dtype
            state[opt_param][name] = np.zeros(shape, dtype = dtype)
    state['t'] = 0

    return state

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


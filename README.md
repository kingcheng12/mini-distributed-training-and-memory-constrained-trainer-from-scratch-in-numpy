# Mini Distributed Training and Memory-Constrained Trainer from Scratch in NumPy

Build a complete training stack in pure NumPy that mirrors how modern frameworks scale models and fit them into limited memory. Implement an MLP with manual autograd, then add gradient accumulation, activation checkpointing, mixed precision, data-parallel all-reduce, and ZeRO-style optimizer sharding under realistic memory budgets.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** make_synthetic_regression_batch
- [x] **2.** init_mlp_params
- [x] **3.** linear_forward
- [x] **4.** relu_forward
- [x] **5.** mlp_forward
- [x] **6.** mse_loss_and_grad
- [x] **7.** linear_backward
- [x] **8.** relu_backward
- [x] **9.** first_linear_backward
- [x] **10.** mlp_backward
- [x] **11.** split_into_micro_batches
- [x] **12.** accumulate_gradients
- [x] **13.** scale_accumulated_gradients
- [x] **14.** grad_accumulation_step
- [x] **15.** mlp_forward_checkpointed
- [x] **16.** recompute_block_activations
- [x] **17.** mlp_backward_checkpointed
- [x] **18.** estimate_checkpointing_memory_savings
- [x] **19.** cast_to_half_precision
- [x] **20.** make_master_params
- [x] **21.** scale_loss
- [x] **22.** unscale_gradients
- [x] **23.** has_non_finite_gradients
- [x] **24.** mixed_precision_step
- [x] **25.** shard_dataset_across_workers
- [x] **26.** compute_local_gradients
- [x] **27.** all_reduce_mean
- [x] **28.** ring_all_reduce_mean
- [x] **29.** data_parallel_train_step
- [x] **30.** bucket_gradients
- [ ] **31.** init_adam_state
- [ ] **32.** partition_optimizer_state
- [ ] **33.** local_shard_adam_update
- [ ] **34.** all_gather_param_shards
- [ ] **35.** zero_optimizer_step
- [ ] **36.** compute_param_memory_bytes
- [ ] **37.** compute_optimizer_memory_bytes
- [ ] **38.** compute_peak_activation_memory_bytes
- [ ] **39.** compare_memory_with_and_without_optimizations
- [ ] **40.** full_distributed_training_loop

---

Built on Deep-ML.

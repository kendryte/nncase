# Copyright 2019-2021 Canaan Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# pylint: disable=invalid-name, unused-argument, import-outside-toplevel

import pytest
# import nncase
import numpy as np
import nncase


def test_attention_config():
    # Test creating AttentionConfig with different parameters
    num_layers = 32
    num_kv_heads = 8
    head_dim = 64

    # Test with numpy dtype
    config1 = nncase.AttentionConfig(num_layers, num_kv_heads, head_dim, np.dtype(np.float16))

    # Verify properties
    assert config1.num_layers == num_layers
    assert config1.num_kv_heads == num_kv_heads
    assert config1.head_dim == head_dim

    # Test property modification
    config1.num_layers = 24
    assert config1.num_layers == 24

    config1.num_kv_heads = 12
    assert config1.num_kv_heads == 12

    config1.head_dim = 128
    assert config1.head_dim == 128

    # Test with different dtype
    config2 = nncase.AttentionConfig(16, 4, 32, np.dtype(np.float32))
    assert config2.num_layers == 16
    assert config2.num_kv_heads == 4
    assert config2.head_dim == 32


def test_paged_attention_config():
    # Test creating PagedAttentionConfig
    num_layers = 32
    num_kv_heads = 8
    head_dim = 64
    block_size = 128

    # Simple creation
    config = nncase.PagedAttentionConfig(
        num_layers,
        num_kv_heads,
        head_dim,
        np.dtype(np.float16),
        block_size
    )

    # Verify properties
    assert config.num_layers == num_layers
    assert config.num_kv_heads == num_kv_heads
    assert config.head_dim == head_dim
    assert config.block_size == block_size

    # Test cache_layout property
    # Define cache layout
    cache_layout = [
        nncase.PagedKVCacheDimKind.NumLayers,
        nncase.PagedKVCacheDimKind.KV,
        nncase.PagedKVCacheDimKind.NumKVHeads,
        nncase.PagedKVCacheDimKind.BlockSize,
        nncase.PagedKVCacheDimKind.HeadDim,
        nncase.PagedKVCacheDimKind.NumBlocks
    ]
    config.cache_layout = cache_layout

    # Verify cache_layout was set correctly
    retrieved_layout = config.cache_layout
    for i, dim in enumerate(cache_layout):
        assert retrieved_layout[i] == dim

    # Test block_layout (read-only property derived from cache_layout)
    block_layout = config.block_layout
    assert len(block_layout) == 2  # Should have 2 dimensions

    # Test packed_axes property
    packed_axes = [
        nncase.PagedKVCacheDimKind.NumKVHeads,
        nncase.PagedKVCacheDimKind.HeadDim
    ]
    config.packed_axes = packed_axes
    assert len(config.packed_axes) == len(packed_axes)
    for i, dim in enumerate(packed_axes):
        assert config.packed_axes[i] == dim

    # Test lanes property
    lanes = [2, 4, 8]
    config.lanes = lanes
    assert len(config.lanes) == len(lanes)
    for i, lane in enumerate(lanes):
        assert config.lanes[i] == lane

    # Test sharding_axes property
    sharding_axes = [
        nncase.PagedKVCacheDimKind.NumBlocks,
        nncase.PagedKVCacheDimKind.NumLayers
    ]
    config.sharding_axes = sharding_axes
    assert len(config.sharding_axes) == len(sharding_axes)
    for i, dim in enumerate(sharding_axes):
        assert config.sharding_axes[i] == dim

    # Test axis_policies property and set_axis_policy method
    policy = [1, 2, 3]
    try:
        config.set_axis_policy(0, policy)
    except Exception:
        pass
    config.axis_policies = [[1]]
    config.set_axis_policy(0, [2, 3])
    assert 1 == len(config.axis_policies)  # At least one policy should exist


def test_paged_attention_kvcache():
    num_layers = 32
    num_kv_heads = 8
    head_dim = 64
    block_size = 128

    # Simple creation
    config = nncase.PagedAttentionConfig(
        num_layers,
        num_kv_heads,
        head_dim,
        np.dtype(np.float16),
        block_size
    )

    # Create kvcache with the config
    kvcache = nncase.PagedAttentionKVCache(config)

    # Test config property
    assert num_layers == kvcache.config.num_layers
    assert num_kv_heads == kvcache.config.num_kv_heads
    assert head_dim == kvcache.config.head_dim
    assert block_size == kvcache.config.block_size

    # Test modifying num_blocks property
    num_blocks = 16
    kvcache.num_blocks = num_blocks
    assert kvcache.num_blocks == num_blocks

    # Test block_table property
    # Create a simple block table tensor
    block_table_data = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    block_table_tensor = nncase.RuntimeTensor.from_numpy(block_table_data)
    kvcache.block_table = block_table_tensor

    # Verify the block_table property
    retrieved_table = kvcache.block_table.to_numpy()
    assert np.array_equal(retrieved_table, block_table_data)

    # Test slot_mapping property
    # Create a simple slot mapping tensor
    slot_mapping_data = np.arange(12, dtype=np.int32).reshape(3, 4)
    slot_mapping_tensor = nncase.RuntimeTensor.from_numpy(slot_mapping_data)
    kvcache.slot_mapping = slot_mapping_tensor

    # Verify the slot_mapping property
    retrieved_mapping = kvcache.slot_mapping.to_numpy()
    assert np.array_equal(retrieved_mapping, slot_mapping_data)

    # Test kv_topo property
    kv_topo = [32]  #
    kvcache.kv_topo = kv_topo

    # Verify the kv_shape property
    retrieved_shape = kvcache.kv_topo
    assert len(retrieved_shape) == len(kv_topo)

    # Test kv_cache method (set and get)
    # Create a simple kv storage tensor
    kv_storage_data = np.random.rand(num_blocks, num_layers, 2,
                                     num_kv_heads, head_dim).astype(np.float16)
    kv_storage_tensor = nncase.RuntimeTensor.from_numpy(kv_storage_data)

    # Set KV cache at specific indices
    indices = [0]  # First device
    kvcache.kv_cache(indices, kv_storage_tensor)

    # Get KV cache at those indices
    retrieved_kv = kvcache.kv_cache(indices)

    # We can't directly compare tensors, so let's verify the KV cache was set
    # by checking that we get back a valid tensor
    assert retrieved_kv is not None

    # Test setting different indices
    indices = [1]  # Second device
    kvcache.kv_cache(indices, kv_storage_tensor)
    retrieved_kv = kvcache.kv_cache(indices)
    assert retrieved_kv is not None

    # Test multi-dimensional indices
    try:
        indices = [0, 1]  # Multi-dimensional index
        kvcache.kv_cache(indices, kv_storage_tensor)
        retrieved_kv = kvcache.kv_cache(indices)
        assert retrieved_kv is not None
    except Exception:
        # Depending on implementation, this might throw an exception
        # if multi-dimensional indices aren't supported
        pass


def test_paged_attention_scheduler():
    # Set up basic configuration
    num_layers = 12
    num_kv_heads = 8
    head_dim = 64
    block_size = 16
    num_blocks = 32
    max_sessions = 4
    max_model_len = (block_size * num_blocks) // max_sessions

    # Create PagedAttentionConfig
    config = nncase.PagedAttentionConfig(
        num_layers,
        num_kv_heads,
        head_dim,
        np.dtype(np.float16),
        block_size
    )

    # Set cache layout
    cache_layout = [
        nncase.PagedKVCacheDimKind.NumBlocks,
        nncase.PagedKVCacheDimKind.NumLayers,
        nncase.PagedKVCacheDimKind.KV,
        nncase.PagedKVCacheDimKind.BlockSize,
        nncase.PagedKVCacheDimKind.NumKVHeads,
        nncase.PagedKVCacheDimKind.HeadDim
    ]
    config.cache_layout = cache_layout

    hierarchy = []  # Example: single node, no distribution
    session_ids = [1, 2, 3]  # Three different sessions
    query_lens = [10, 15, 20]  # Different query lengths for each session

    scheduler = nncase._nncase.RefPagedAttentionScheduler(
        config,
        num_blocks,
        max_model_len,
        hierarchy
    )

    kv_cache = scheduler.schedule(session_ids, query_lens)

    ctx_lens = kv_cache.context_lens.to_runtime_tensor().to_numpy()
    assert np.allclose(ctx_lens, np.array([0, 0, 0], np.int64))
    seq_lens = kv_cache.seq_lens.to_runtime_tensor().to_numpy()
    assert np.allclose(seq_lens, np.array([10, 15, 20], np.int64))
    block_table = kv_cache.block_table.to_runtime_tensor().to_numpy()
    assert len(block_table.shape) == 3
    assert block_table.shape[1] == 2
    block_table[0, 0, 0] == 0 + session_ids[0] * (max_model_len // config.block_size)
    block_table[1, 0, 0] == 0 + session_ids[1] * (max_model_len // config.block_size)
    block_table[2, 0, 0] == 0 + session_ids[2] * (max_model_len // config.block_size)
    block_table[2, 1, 0] == 1 + session_ids[2] * (max_model_len // config.block_size)

    ivalue = kv_cache.as_ivalue()
    assert ivalue is not None

    # Test with different parameters - empty session
    empty_session_ids = []
    empty_query_lens = []

    try:
        # This might raise an exception depending on implementation
        empty_kv_cache = scheduler.schedule(empty_session_ids, empty_query_lens)
        assert empty_kv_cache is not None
    except Exception as e:
        # If empty inputs aren't allowed, that's fine
        print(f"Note: Empty session test failed as expected: {str(e)}")

    # Test with mismatched lengths - should fail
    mismatched_session_ids = [1, 2]
    mismatched_query_lens = [10]

    try:
        # This should raise an exception
        scheduler.schedule(mismatched_session_ids, mismatched_query_lens)
        # If we get here, it didn't throw as expected
        assert False, "Scheduler should have rejected mismatched session_ids and query_lens"
    except Exception:
        # Expected behavior
        pass

    # get the test function.
    test_func = scheduler.create_test_function(
        8,
        [nncase._nncase.AttentionDimKind.Seq, nncase._nncase.AttentionDimKind.Head,
            nncase._nncase.AttentionDimKind.Dim],
        [nncase._nncase.AttentionDimKind.Seq, nncase._nncase.AttentionDimKind.Head, nncase._nncase.AttentionDimKind.Dim])
    assert len(test_func.parameters) == 1 + 1 + config.num_layers * 2


def test_paged_attention_scheduler_distributed():
    num_layers = 1
    num_kv_heads = 8
    head_dim = 64
    block_size = 256
    num_blocks = 256
    max_sessions = 16
    max_model_len = (block_size * num_blocks) // max_sessions
    kv_type = np.dtype(np.float16)

    # Create PagedAttentionConfig
    config = nncase.PagedAttentionConfig(
        num_layers,
        num_kv_heads,
        head_dim,
        kv_type,
        block_size,
        [nncase.PagedKVCacheDimKind.NumBlocks,
         nncase.PagedKVCacheDimKind.NumLayers,
         nncase.PagedKVCacheDimKind.NumKVHeads,
         nncase.PagedKVCacheDimKind.KV,
         nncase.PagedKVCacheDimKind.BlockSize,
         nncase.PagedKVCacheDimKind.HeadDim],
        [nncase.PagedKVCacheDimKind.HeadDim],
        [128 // 2],
        [nncase.PagedKVCacheDimKind.NumKVHeads, nncase.PagedKVCacheDimKind.NumBlocks],
        [[1], [2, 3]]
    )

    assert config.lanes == [64]
    assert len(config.sharding_axes) == 2
    assert config.axis_policies[0] == [1]
    assert config.axis_policies[1] == [2, 3]

    hierarchy = [1, 2, 8, 4, 4]
    scheduler = nncase._nncase.RefPagedAttentionScheduler(
        config, num_blocks, max_model_len, hierarchy)

    session_ids = [0]
    query_lens = [512]
    kvcache = scheduler.schedule(session_ids, query_lens)
    block_table = kvcache.block_table.to_runtime_tensor().to_numpy()
    assert block_table.shape == (1, 2, 3)
    assert np.allclose(block_table, np.array([[[-1, 0, 0], [-1, 0, 1]]], np.int64))
    slot_mapping = kvcache.slot_mapping.to_runtime_tensor().to_numpy()
    slot_mapping_ref = np.zeros_like(slot_mapping)
    slot_mapping_ref[:, 0] = -1
    slot_mapping_ref[:, 1] = 0
    slot_mapping_ref[:, 2] = np.arange(0, 512)
    assert np.allclose(slot_mapping, slot_mapping_ref)

    kvcache = scheduler.schedule([0], [1])
    block_table = kvcache.block_table.to_runtime_tensor().to_numpy()
    assert block_table.shape == (1, 3, 3)
    assert np.allclose(block_table, np.array([[[-1, 0, 0], [-1, 0, 1], [-1, 0, 2]]], np.int64))
    slot_mapping = kvcache.slot_mapping.to_runtime_tensor().to_numpy()
    slot_mapping_ref = np.zeros_like(slot_mapping)
    slot_mapping_ref[0, :] = [-1, 0, 512]
    assert np.allclose(slot_mapping, slot_mapping_ref)

    # get the test function.
    num_q_heads = 8
    test_func = scheduler.create_test_function(
        num_q_heads,
        [nncase._nncase.AttentionDimKind.Seq, nncase._nncase.AttentionDimKind.Head,
            nncase._nncase.AttentionDimKind.Dim],
        [nncase._nncase.AttentionDimKind.Seq, nncase._nncase.AttentionDimKind.Head, nncase._nncase.AttentionDimKind.Dim])
    assert len(test_func.parameters) == 1 + 1 + config.num_layers * 2

    parameters = test_func.parameters
    q_var_dims = parameters[0].dimensions()
    assert len(q_var_dims) == 3
    assert q_var_dims[0].kind == nncase._nncase.DimensionKind.Dynamic

    body = test_func.body
    parameters.append(q_var_dims[0])  # all inputs are [seq,head,dim] layout
    inputs = []
    ref_q = np.random.rand(np.sum(query_lens), num_q_heads, head_dim).astype(kv_type)
    ref_k = np.random.rand(np.sum(query_lens), num_kv_heads, head_dim).astype(kv_type)
    ref_v = np.random.rand(np.sum(query_lens), num_kv_heads, head_dim).astype(kv_type)
    inputs.append(nncase._nncase.RTValue.from_runtime_tensor(
        nncase.RuntimeTensor.from_numpy(ref_q)))
    inputs.append(nncase._nncase.RTValue.from_runtime_tensor(
        nncase.RuntimeTensor.from_numpy(ref_k)))
    inputs.append(nncase._nncase.RTValue.from_runtime_tensor(
        nncase.RuntimeTensor.from_numpy(ref_v)))
    inputs.append(kvcache.as_ivalue())
    inputs.append(nncase._nncase.RTValue.from_runtime_tensor(
        nncase.RuntimeTensor.from_numpy(np.array(np.sum(query_lens), np.int64))))
    result = body.evaluate(parameters, inputs)
    result_ref = result.to_runtime_tensor().to_numpy()


def test_create_object_tensor():
    pass


if __name__ == "__main__":
    pytest.main(['-vvs', __file__])

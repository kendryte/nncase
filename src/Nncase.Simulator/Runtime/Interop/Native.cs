// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime.Interop;

internal static class Native
{
    public const string LibraryName = "Nncase.Runtime.Native";

    [DllImport(LibraryName, EntryPoint = "nncase_object_add_ref")]
    public static extern ErrorCode ObjectAddRef(IntPtr obj);

    [DllImport(LibraryName, EntryPoint = "nncase_object_release")]
    public static extern ErrorCode ObjectRelease(IntPtr obj);

    [DllImport(LibraryName, EntryPoint = "nncase_interp_create")]
    public static extern ErrorCode InterpCreate(out RTInterpreter interp);

    [DllImport(LibraryName, EntryPoint = "nncase_interp_free")]
    public static extern ErrorCode InterpFree(IntPtr interp);

    [DllImport(LibraryName, EntryPoint = "nncase_interp_load_model")]
    public static extern unsafe ErrorCode InterpLoadModel(RTInterpreter interp, void* modelBuffer, uint modelSize, bool copyBuffer);

    [DllImport(LibraryName, EntryPoint = "nncase_interp_load_model_from_path")]
    public static extern unsafe ErrorCode InterpLoadModel(RTInterpreter interp, [MarshalAs(UnmanagedType.LPStr)] string modelBuffer);

    [DllImport(LibraryName, EntryPoint = "nncase_interp_set_dump_root", CharSet = CharSet.Ansi)]
    public static extern unsafe ErrorCode InterpSetDumpRoot(RTInterpreter interp, [MarshalAs(UnmanagedType.LPStr)] string path);

    [DllImport(LibraryName, EntryPoint = "nncase_interp_get_entry_func")]
    public static extern unsafe ErrorCode InterpGetEntryFunction(RTInterpreter interp, out IntPtr func);

    [DllImport(LibraryName, EntryPoint = "nncase_func_get_params_size")]
    public static extern unsafe ErrorCode FuncGetParamsSize(IntPtr func, out uint size);

    [DllImport(LibraryName, EntryPoint = "nncase_func_invoke")]
    public static extern unsafe ErrorCode FuncInvoke(IntPtr func, IntPtr* @params, uint paramsSize, out IntPtr result);

    [DllImport(LibraryName, EntryPoint = "nncase_buffer_allocator_get_host")]
    public static extern ErrorCode BufferAllocatorGetHost(out IntPtr allocator);

    [DllImport(LibraryName, EntryPoint = "nncase_buffer_allocator_alloc")]
    public static extern unsafe ErrorCode BufferAllocatorAlloc(IntPtr allocator, uint bytes, void* options, out IntPtr buffer);

    [DllImport(LibraryName, EntryPoint = "nncase_buffer_as_host")]
    public static extern unsafe ErrorCode BufferAsHost(IntPtr buffer, out IntPtr hostBuffer);

    [DllImport(LibraryName, EntryPoint = "nncase_host_buffer_map")]
    public static extern ErrorCode HostBufferMap(IntPtr hostBuffer, RTMapAccess mapAccess, out IntPtr data, out uint bytes);

    [DllImport(LibraryName, EntryPoint = "nncase_host_buffer_unmap")]
    public static extern unsafe ErrorCode HostBufferUnmap(IntPtr hostBuffer);

    [DllImport(LibraryName, EntryPoint = "nncase_dtype_create_prime")]
    public static extern unsafe ErrorCode DTypeCreatePrim(TypeCode typeCode, out RTDataType dtype);

    [DllImport(LibraryName, EntryPoint = "nncase_dtype_create_vector")]
    public static extern unsafe ErrorCode DTypeCreateVector(RTDataType elemType, int[] lanes, int length, out RTVectorType dtype);

    [DllImport(LibraryName, EntryPoint = "nncase_dtype_get_typecode")]
    public static extern unsafe TypeCode DTypeGetTypeCode(RTDataType handle);

    [DllImport(LibraryName, EntryPoint = "nncase_vector_dtype_get_elem_type")]
    public static extern unsafe ErrorCode VectorDTypeGetElemType(RTVectorType handle, out RTDataType elemType);

    [DllImport(LibraryName, EntryPoint = "nncase_vector_dtype_get_lanes_length")]
    public static extern unsafe ErrorCode VectorDTypeGetLanesLength(RTVectorType handle, out int length);

    [DllImport(LibraryName, EntryPoint = "nncase_vector_dtype_get_lanes")]
    public static extern unsafe ErrorCode VectorDTypeGetLanes(RTVectorType handle, [Out] int[] lanes);

    [DllImport(LibraryName, EntryPoint = "nncase_dtype_create_reference")]
    public static extern unsafe ErrorCode DTypeCreateReference(RTDataType elemType, out RTDataType dtype);

    [DllImport(LibraryName, EntryPoint = "nncase_dtype_create_attention_kv_cache")]
    public static extern unsafe ErrorCode DTypeCreateAttentionKVCache(out RTDataType dtype);

    [DllImport(LibraryName, EntryPoint = "nncase_dtype_create_paged_attention_kv_cache")]
    public static extern unsafe ErrorCode DTypeCreatePagedAttentionKVCache(out RTDataType dtype);

    [DllImport(LibraryName, EntryPoint = "nncase_value_is_tensor")]
    public static extern unsafe ErrorCode ValueIsTensor(IntPtr value, out bool isTensor);

    [DllImport(LibraryName, EntryPoint = "nncase_tensor_create")]
    public static extern unsafe ErrorCode TensorCreate(RTDataType dtype, uint* dims, uint dimsLength, uint* strides, uint stridesLength, in RTBufferSlice.RuntimeStruct bufferSlice, out RTTensor tensor);

    [DllImport(LibraryName, EntryPoint = "nncase_tensor_get_dtype")]
    public static extern unsafe ErrorCode TensorGetDtype(RTTensor tensor, out RTDataType dtype);

    [DllImport(LibraryName, EntryPoint = "nncase_tensor_get_buffer")]
    public static extern unsafe ErrorCode TensorGetBuffer(RTTensor tensor, out RTBufferSlice.RuntimeStruct buffer);

    [DllImport(LibraryName, EntryPoint = "nncase_tensor_get_dims")]
    public static extern unsafe ErrorCode TensorGetDims(RTTensor tensor, uint* dims, ref uint dimsLength);

    [DllImport(LibraryName, EntryPoint = "nncase_tensor_get_strides")]
    public static extern unsafe ErrorCode TensorGetStrides(RTTensor tensor, uint* strides, ref uint stridesLength);

    [DllImport(LibraryName, EntryPoint = "nncase_tuple_create")]
    public static extern unsafe ErrorCode TupleCreate(IntPtr* fields, uint fieldsLength, out RTTuple tuple);

    [DllImport(LibraryName, EntryPoint = "nncase_tuple_get_fields")]
    public static extern unsafe ErrorCode TupleGetFields(RTTuple tuple, IntPtr* fields, ref uint fieldsLength);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_config_create")]
    public static extern ErrorCode AttentionConfigCreate(int num_layers, int num_kv_heads, int head_dim, TypeCode kv_type, out RTAttentionConfig config);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_config_get_num_layers")]
    public static extern ErrorCode AttentionConfigGetNumLayers(RTAttentionConfig config, out int num_layers);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_config_set_num_layers")]
    public static extern ErrorCode AttentionConfigSetNumLayers(RTAttentionConfig config, int num_layers);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_config_get_num_kv_heads")]
    public static extern ErrorCode AttentionConfigGetNumKvHeads(RTAttentionConfig config, out int num_kv_heads);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_config_set_num_kv_heads")]
    public static extern ErrorCode AttentionConfigSetNumKvHeads(RTAttentionConfig config, int num_kv_heads);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_config_get_head_dim")]
    public static extern ErrorCode AttentionConfigGetHeadDim(RTAttentionConfig config, out int head_dim);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_config_set_head_dim")]
    public static extern ErrorCode AttentionConfigSetHeadDim(RTAttentionConfig config, int head_dim);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_config_get_kv_type")]
    public static extern ErrorCode AttentionConfigGetKvType(RTAttentionConfig config, out TypeCode kv_type);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_config_set_kv_type")]
    public static extern ErrorCode AttentionConfigSetKvType(RTAttentionConfig config, TypeCode kv_type);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_config_create")]
    public static extern ErrorCode PagedAttentionConfigCreate(
        int num_layers,
        int num_kv_heads,
        int head_dim,
        TypeCode kv_type,
        int block_size,
        [In] IR.NN.PagedKVCacheDimKind[] cache_layout,
        [In] IR.NN.PagedKVCacheDimKind[] packed_axes,
        int packed_axes_len,
        [In] int[] lanes,
        int lanes_len,
        [In] IR.NN.PagedKVCacheDimKind[] sharding_axes,
        int sharding_axes_len,
        [In] int[] axis_policies,
        [In] int[] axis_policies_lens,
        out RTPagedAttentionConfig config);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_config_get_block_size")]
    public static extern ErrorCode PagedAttentionConfigGetBlockSize(RTPagedAttentionConfig config, out int block_size);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_config_set_block_size")]
    public static extern ErrorCode PagedAttentionConfigSetBlockSize(RTPagedAttentionConfig config, int block_size);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_config_get_cache_layout")]
    public static extern ErrorCode PagedAttentionConfigGetCacheLayout(RTPagedAttentionConfig config, [Out] IR.NN.PagedKVCacheDimKind[] layout, int layout_len);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_config_set_cache_layout")]
    public static extern ErrorCode PagedAttentionConfigSetCacheLayout(RTPagedAttentionConfig config, [In] IR.NN.PagedKVCacheDimKind[] layout, int layout_len);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_config_get_packed_axes")]
    public static extern ErrorCode PagedAttentionConfigGetPackedAxes(RTPagedAttentionConfig config, [Out] IR.NN.PagedKVCacheDimKind[] packed_axes, int packed_axes_len);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_config_set_packed_axes")]
    public static extern ErrorCode PagedAttentionConfigSetPackedAxes(RTPagedAttentionConfig config, [In] IR.NN.PagedKVCacheDimKind[] packed_axes, int packed_axes_len);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_config_get_lanes")]
    public static extern ErrorCode PagedAttentionConfigGetLanes(RTPagedAttentionConfig config, [Out] int[] lanes, int lanes_len);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_config_set_lanes")]
    public static extern ErrorCode PagedAttentionConfigSetLanes(RTPagedAttentionConfig config, [In] int[] lanes, int lanes_len);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_config_get_sharding_axes")]
    public static extern ErrorCode PagedAttentionConfigGetShardingAxes(
        RTPagedAttentionConfig config,
        [Out] IR.NN.PagedKVCacheDimKind[] sharding_axes,
        int sharding_axes_len);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_config_set_sharding_axes")]
    public static extern ErrorCode PagedAttentionConfigSetShardingAxes(
        RTPagedAttentionConfig config,
        [In] IR.NN.PagedKVCacheDimKind[] sharding_axes,
        int sharding_axes_len);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_config_get_axis_policy_len")]
    public static extern ErrorCode PagedAttentionConfigGetAxisPolicyLen(
        RTPagedAttentionConfig config,
        int i,
        out int policy_len);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_config_get_axis_policy")]
    public static extern ErrorCode PagedAttentionConfigGetAxisPolicy(
        RTPagedAttentionConfig config,
        int i,
        [Out] int[] axis_policy,
        int axis_policy_len);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_config_set_axis_policy")]
    public static extern ErrorCode PagedAttentionConfigSetAxisPolicy(
        RTPagedAttentionConfig config,
        int i,
        [In] int[] axis_policy,
        int axis_policy_len);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_kv_cache_create")]
    public static extern ErrorCode AttentionKVCacheCreate(
        RTAttentionConfig config,
        int num_seqs,
        int num_tokens,
        RTTensor context_lens,
        RTTensor seq_lens,
        out RTAttentionKVCache cache);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_kv_cache_get_config")]
    public static extern ErrorCode AttentionKVCacheGetConfig(
        RTAttentionKVCache cache,
        out RTAttentionConfig config);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_kv_cache_get_num_seqs")]
    public static extern ErrorCode AttentionKVCacheGetNumSeqs(
        RTAttentionKVCache cache,
        out int num_seqs);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_kv_cache_set_num_seqs")]
    public static extern ErrorCode AttentionKVCacheSetNumSeqs(
        RTAttentionKVCache cache,
        int num_seqs);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_kv_cache_get_num_tokens")]
    public static extern ErrorCode AttentionKVCacheGetNumTokens(
        RTAttentionKVCache cache,
        out int num_tokens);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_kv_cache_set_num_tokens")]
    public static extern ErrorCode AttentionKVCacheSetNumTokens(
        RTAttentionKVCache cache,
        int num_tokens);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_kv_cache_create")]
    public static extern ErrorCode PagedAttentionKVCacheCreate(
        RTPagedAttentionConfig config,
        int num_seqs,
        int num_tokens,
        RTTensor context_lens,
        RTTensor seq_lens,
        RTTensor block_table,
        RTTensor slot_mapping,
        int num_blocks,
        [In] int[] kv_shape,
        int kv_shape_len,
        out RTPagedAttentionKVCache cache);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_kv_cache_get_num_blocks")]
    public static extern ErrorCode PagedAttentionKVCacheGetNumBlocks(
        RTPagedAttentionKVCache cache,
        out int num_blocks);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_kv_cache_set_kv_cache")]
    public static extern ErrorCode PagedAttentionKVCacheSetKVCache(
        RTPagedAttentionKVCache cache,
        int[] indices,
        int indices_len,
        RTTensor kv_cache);

    [DllImport(LibraryName, EntryPoint = "nncase_wait_for_debugger")]
    public static extern int NncaseWaitForDebugger(byte enable);

    [DllImport(LibraryName, EntryPoint = "nncase_continue_execution")]
    public static extern int NncaseContinueExecution();
}

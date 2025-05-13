﻿// Copyright (c) Canaan Inc. All rights reserved.
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
    public static extern unsafe ErrorCode AttentionConfigCreate(int num_layers, int num_kv_heads, int head_dim, TypeCode kv_type, out RTAttentionConfig config);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_config_get_num_layers")]
    public static extern unsafe ErrorCode AttentionConfigGetNumLayers(RTAttentionConfig config, ref int num_layers);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_config_set_num_layers")]
    public static extern unsafe ErrorCode AttentionConfigSetNumLayers(RTAttentionConfig config, int num_layers);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_config_get_num_kv_heads")]
    public static extern unsafe ErrorCode AttentionConfigGetNumKvHeads(RTAttentionConfig config, ref int num_kv_heads);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_config_set_num_kv_heads")]
    public static extern unsafe ErrorCode AttentionConfigSetNumKvHeads(RTAttentionConfig config, int num_kv_heads);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_config_get_head_dim")]
    public static extern unsafe ErrorCode AttentionConfigGetHeadDim(RTAttentionConfig config, ref int head_dim);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_config_set_head_dim")]
    public static extern unsafe ErrorCode AttentionConfigSetHeadDim(RTAttentionConfig config, int head_dim);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_config_get_kv_type")]
    public static extern unsafe ErrorCode AttentionConfigGetKVType(RTAttentionConfig config, out TypeCode kv_type);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_config_set_kv_type")]
    public static extern unsafe ErrorCode AttentionConfigSetKVType(RTAttentionConfig config, TypeCode kv_type);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_config_create")]
    public static extern unsafe ErrorCode PagedAttentionConfigCreate(int num_layers, int num_kv_heads, int head_dim, TypeCode kv_type, int block_size, out RTPagedAttentionConfig config);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_config_get_block_size")]
    public static extern unsafe ErrorCode PagedAttentionConfigGetBlockSize(RTPagedAttentionConfig config, out int block_size);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attention_config_set_block_size")]
    public static extern unsafe ErrorCode PagedAttentionConfigSetBlockSize(RTPagedAttentionConfig config, int block_size);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_kv_cache_get_num_requests")]
    public static extern unsafe ErrorCode AttentionKvCacheGetNumRequests(RTAttentionKVCache kvcache, out int num_requests);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_kv_cache_get_seq_len")]
    public static extern unsafe ErrorCode AttentionKvCacheGetSeqLen(RTAttentionKVCache kvcache, int request_id, out long seq_len);

    [DllImport(LibraryName, EntryPoint = "nncase_attention_kv_cache_get_context_len")]
    public static extern unsafe ErrorCode AttentionKvCacheGetContextLen(RTAttentionKVCache kvcache, int request_id, out long context_len);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attenion_scheduler_create")]
    public static extern unsafe ErrorCode PagedAttentionSchedulerCreate(RTPagedAttentionConfig config, int numBlocks, int maxModelLen, out RTPagedAttentionScheduler scheduler);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attenion_scheduler_schedule")]
    public static extern unsafe ErrorCode PagedAttentionSchedulerSchedule(RTPagedAttentionScheduler rTPagedAttentionScheduler, RTTensor sessionIds, RTTensor tokenCounts, out RTPagedAttentionKVCache cache);

    /*
    [DllImport(LibraryName, EntryPoint = "nncase_paged_attenion_kv_cache_get_block")]
    public static extern unsafe ErrorCode GetBlock(RTPagedAttentionKVCache cache, IR.NN.AttentionCacheKind kind, int layerId, long blockId, out RTTensor tensor);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attenion_kv_cache_get_context_block_ids")]
    public static extern unsafe ErrorCode GetContextBlockIds(RTPagedAttentionKVCache cache, int requestId, out RTTensor tensor);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attenion_kv_cache_get_output_slot_ids")]
    public static extern unsafe ErrorCode GetOutputSlotIds(RTPagedAttentionKVCache cache, out RTTensor tensor);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attenion_kv_cache_get_slot")]
    public static extern unsafe ErrorCode GetSlot(RTPagedAttentionKVCache cache, IR.NN.AttentionCacheKind kind, int layerId, long slotId, out RTTensor tensor);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attenion_kv_cache_get_slots")]
    public static extern unsafe ErrorCode GetSlots(RTPagedAttentionKVCache cache, [In] RTTensor block, int startSlot, int count, out RTTensor tensor);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attenion_kv_cache_update_output_slot")]
    public static extern unsafe ErrorCode UpdateOutputSlot(RTPagedAttentionKVCache cache, IR.NN.AttentionCacheKind kind, int layerId, long slotId, [In] RTTensor slot);
    */

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attenion_kv_cache_get_sub_block")]
    public static extern unsafe ErrorCode PagedAttenionKVCacheGetSubBlock(RTPagedAttentionKVCache cache, [In] int[] indices, int indices_len, out RTTensor sub_block);

    [DllImport(LibraryName, EntryPoint = "nncase_paged_attenion_kv_cache_set_sub_block")]
    public static extern unsafe ErrorCode PagedAttenionKVCacheSetSubBlock(RTPagedAttentionKVCache cache, [In] int[] indices, int indices_len, RTTensor sub_block);
}

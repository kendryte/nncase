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

    [DllImport(LibraryName, EntryPoint = "nncase_dtype_get_typecode")]
    public static extern unsafe TypeCode DTypeGetTypeCode(RTDataType handle);

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
}

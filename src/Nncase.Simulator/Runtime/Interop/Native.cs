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

    [DllImport(LibraryName, EntryPoint = "nncase_object_free")]
    public static extern ErrorCode ObjectFree(IntPtr obj);

    [DllImport(LibraryName, EntryPoint = "nncase_interp_create")]
    public static extern ErrorCode InterpCreate(out IntPtr interp);

    [DllImport(LibraryName, EntryPoint = "nncase_interp_free")]
    public static extern ErrorCode InterpFree(IntPtr interp);

    [DllImport(LibraryName, EntryPoint = "nncase_interp_load_model")]
    public static extern unsafe ErrorCode InterpLoadModel(IntPtr interp, void* modelBuffer, uint modelSize, bool copyBuffer);

    [DllImport(LibraryName, EntryPoint = "nncase_interp_get_entry_func")]
    public static extern unsafe ErrorCode InterpGetEntryFunction(IntPtr interp, out IntPtr func);

    [DllImport(LibraryName, EntryPoint = "nncase_func_get_params_size")]
    public static extern unsafe ErrorCode FuncGetParamsSize(IntPtr func, out uint size);

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
    public static extern unsafe ErrorCode DTypeCreatePrim(TypeCode typeCode, out IntPtr dtype);
}

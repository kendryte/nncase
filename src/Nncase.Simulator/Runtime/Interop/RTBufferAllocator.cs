// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime.Interop;

/// <summary>
/// Runtime buffer allocator.
/// </summary>
public class RTBufferAllocator
{
    private static RTBufferAllocator? _host;
    private readonly IntPtr _handle;

    internal RTBufferAllocator(IntPtr handle)
    {
        _handle = handle;
    }

    /// <summary>
    /// Gets host buffer allocator.
    /// </summary>
    public static RTBufferAllocator Host
    {
        get
        {
            if (_host == null)
            {
                Native.BufferAllocatorGetHost(out var allocator).ThrowIfFailed();
                _host = new RTBufferAllocator(allocator);
            }

            return _host;
        }
    }

    /// <summary>
    /// Gets handle.
    /// </summary>
    public IntPtr Handle => _handle;

    /// <summary>
    /// Allocate buffer.
    /// </summary>
    /// <param name="bytes">Size in bytes.</param>
    /// <returns>Allocated buffer.</returns>
    public unsafe RTBuffer Allocate(uint bytes)
    {
        Native.BufferAllocatorAlloc(_handle, bytes, null, out var buffer).ThrowIfFailed();
        return new RTBuffer(buffer);
    }
}

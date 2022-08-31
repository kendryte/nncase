// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Buffers;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime.Interop;

/// <summary>
/// Runtime buffer.
/// </summary>
public class RTBuffer : RTObject
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RTBuffer"/> class.
    /// </summary>
    /// <param name="handle">Buffer handle.</param>
    internal RTBuffer(IntPtr handle)
        : base(handle)
    {
    }

    /// <inheritdoc/>
    public override bool IsInvalid => handle == IntPtr.Zero;

    /// <summary>
    /// As host buffer.
    /// </summary>
    /// <returns>Host buffer.</returns>
    public RTHostBuffer? AsHost()
    {
        if (Native.BufferAsHost(handle, out var hostBuffer).IsSuccess)
        {
            return new RTHostBuffer(hostBuffer);
        }

        return null;
    }
}

public enum RTMapAccess
{
    None = 0,
    Read = 1,
    Write = 2,
    ReadWrite = 3,
}

/// <summary>
/// Runtime host buffer.
/// </summary>
public class RTHostBuffer : RTBuffer
{
    /// <summary>
    /// Initializes a new instance of the <see cref="RTHostBuffer"/> class.
    /// </summary>
    /// <param name="handle">Host buffer handle.</param>
    internal RTHostBuffer(IntPtr handle)
        : base(handle)
    {
    }

    /// <summary>
    /// Map host buffer.
    /// </summary>
    /// <param name="mapAccess">Access rights.</param>
    /// <returns>Mapped memory.</returns>
    public unsafe IMemoryOwner<byte> Map(RTMapAccess mapAccess)
    {
        Native.HostBufferMap(handle, mapAccess, out var data, out var bytes).ThrowIfFailed();
        return new RTHostMemoryManager(this, data, bytes);
    }
}

/// <summary>
/// Runtime buffer slice.
/// </summary>
public struct RTBufferSlice
{
    /// <summary>
    /// Buffer.
    /// </summary>
    public RTBuffer Buffer { get; set; }

    /// <summary>
    /// Start.
    /// </summary>
    public uint Start { get; set; }

    /// <summary>
    /// Size in bytes.
    /// </summary>
    public uint SizeBytes { get; set; }

    [StructLayout(LayoutKind.Sequential)]
    internal struct RuntimeStruct
    {
        public IntPtr Buffer;
        public uint Start;
        public uint SizeBytes;
    }

    internal RuntimeStruct ToRT()
    {
        return new RuntimeStruct
        {
            Buffer = Buffer.DangerousGetHandle(),
            Start = Start,
            SizeBytes = SizeBytes
        };
    }

    internal static RTBufferSlice FromRT(in RuntimeStruct rt)
    {
        return new RTBufferSlice
        {
            Buffer = new RTBuffer(rt.Buffer),
            Start = rt.Start,
            SizeBytes = rt.SizeBytes
        };
    }
}

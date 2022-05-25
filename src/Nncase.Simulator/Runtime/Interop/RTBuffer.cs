// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
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

    /// <summary>
    /// As host buffer.
    /// </summary>
    /// <returns>Host buffer.</returns>
    public RTHostBuffer? AsHost()
    {
        if (Native.BufferAsHost(Handle, out var hostBuffer).IsSuccess)
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

    public Memory<byte> Map(RTMapAccess mapAccess)
    {
        throw new NotImplementedException();
    }
}

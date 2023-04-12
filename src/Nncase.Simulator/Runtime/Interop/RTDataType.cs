// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime.Interop;

/// <summary>
/// Runtime data type.
/// </summary>
public class RTDataType : RTObject
{
    internal RTDataType()
        : base(IntPtr.Zero)
    {
    }

    internal RTDataType(IntPtr handle)
        : base(handle)
    {
    }

    /// <summary>
    /// Gets get the type code.
    /// </summary>
    public TypeCode TypeCode => Native.DTypeGetTypeCode(this);

    /// <inheritdoc/>
    public override bool IsInvalid => handle == IntPtr.Zero;

    /// <summary>
    /// Create datatype from typecode.
    /// </summary>
    /// <param name="typeCode">Type code.</param>
    /// <returns>Created datatype.</returns>
    public static RTDataType FromTypeCode(TypeCode typeCode)
    {
        Native.DTypeCreatePrim(typeCode, out var dtype).ThrowIfFailed();
        return dtype;
    }
}

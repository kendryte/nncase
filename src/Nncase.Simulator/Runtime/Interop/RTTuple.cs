// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Buffers;

namespace Nncase.Runtime.Interop;

/// <summary>
/// Runtime tuple.
/// </summary>
public class RTTuple : RTValue
{
    private RTValue[]? _fields;

    internal RTTuple()
        : base(IntPtr.Zero)
    {
    }

    internal RTTuple(IntPtr handle, bool addRef = false)
        : base(handle)
    {
        if (addRef)
        {
            Native.ObjectAddRef(handle);
        }
    }

    /// <summary>
    /// Gets fields.
    /// </summary>
    public unsafe RTValue[] Fields
    {
        get
        {
            if (_fields == null)
            {
                uint fieldsLength = 0;
                Native.TupleGetFields(this, null, ref fieldsLength);
                var fields = new IntPtr[fieldsLength];
                fixed (IntPtr* fieldsPtr = fields)
                {
                    Native.TupleGetFields(this, fieldsPtr, ref fieldsLength).ThrowIfFailed();
                    _fields = fields.Select(value => RTValue.FromHandle(value)).ToArray();
                }
            }

            return _fields;
        }
    }

    /// <inheritdoc/>
    public override bool IsInvalid => handle == IntPtr.Zero;

    public static unsafe RTTuple Create(RTValue[] fields)
    {
        var handles = fields.Select(x => x.DangerousGetHandle()).ToArray();
        fixed (IntPtr* handlesPtr = handles)
        {
            Native.TupleCreate(handlesPtr, (uint)fields.Length, out var tuple).ThrowIfFailed();
            GC.KeepAlive(fields);
            return tuple;
        }
    }

    public static RTTuple FromTuple(TupleValue tv)
    {
        var fields = tv.Select(RTValue.FromValue).ToArray();
        return Create(fields);
    }
}

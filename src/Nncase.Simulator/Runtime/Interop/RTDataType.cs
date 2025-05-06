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

    public static RTDataType From(DataType dataType)
    {
        switch (dataType)
        {
            case PrimType primType:
                return FromTypeCode(primType.TypeCode);
            case ReferenceType referenceType:
                var rType = From(referenceType.ElemType);
                Native.DTypeCreateReference(rType, out var refType).ThrowIfFailed();
                return refType;
            case IR.NN.PagedAttentionKVCacheType:
                Native.DTypeCreatePagedAttentionKVCache(out var pkvType).ThrowIfFailed();
                return pkvType;
            case IR.NN.AttentionKVCacheType:
                Native.DTypeCreateAttentionKVCache(out var kvType).ThrowIfFailed();
                return kvType;
            case VectorType vectorType:
                var elemType = From(vectorType.ElemType);
                var lanes = vectorType.Lanes.ToArray();
                Native.DTypeCreateVector(elemType, lanes, lanes.Length, out var rtDtype).ThrowIfFailed();
                return rtDtype;
            default:
                throw new ArgumentOutOfRangeException(nameof(dataType));
        }
    }
}

public sealed class RTVectorType : RTDataType
{
    internal RTVectorType()
       : base(IntPtr.Zero)
    {
    }

    internal RTVectorType(IntPtr handle)
        : base(handle)
    {
    }

    public RTDataType ElementType
    {
        get
        {
            Native.VectorDTypeGetElemType(this, out var elemType).ThrowIfFailed();
            return elemType;
        }
    }

    public int[] Lanes
    {
        get
        {
            Native.VectorDTypeGetLanesLength(this, out int length).ThrowIfFailed();
            var lanes = new int[length];
            Native.VectorDTypeGetLanes(this, lanes).ThrowIfFailed();
            return lanes;
        }
    }
}

// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime.Interop;

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

    internal static RTDataType FromRTDataType(RTDataType dtype)
    {
        var typecode = Native.DTypeGetTypeCode(dtype);
        var handle = dtype.DangerousGetHandle();
        Native.ObjectAddRef(handle);
        return typecode switch
        {
            TypeCode.Pointer => throw new NotSupportedException(),
            TypeCode.ValueType => new RTValueType(handle),
            TypeCode.VectorType => new RTVectorType(handle),
            TypeCode.ReferenceType => new RTReferenceType(handle),
            _ => new RTPrimType(handle),
        };
    }
}

public sealed class RTPrimType : RTDataType
{
    internal RTPrimType()
       : base(IntPtr.Zero)
    {
    }

    internal RTPrimType(IntPtr handle)
        : base(handle)
    {
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
            return FromRTDataType(elemType);
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

public sealed class RTValueType : RTDataType
{
    internal RTValueType()
       : base(IntPtr.Zero)
    {
    }

    internal RTValueType(IntPtr handle)
        : base(handle)
    {
    }

    public Guid Uuid
    {
        get
        {
            var uuid = new byte[16];
            Native.ValueDTypeGetUUID(this, uuid, uuid.Length).ThrowIfFailed();
            return new Guid(uuid);
        }
    }
}

public sealed class RTReferenceType : RTDataType
{
    internal RTReferenceType()
       : base(IntPtr.Zero)
    {
    }

    internal RTReferenceType(IntPtr handle)
        : base(handle)
    {
    }

    public RTDataType ElemType
    {
        get
        {
            Native.ReferenceDTypeGetElemType(this, out var elemType).ThrowIfFailed();
            return FromRTDataType(elemType);
        }
    }
}

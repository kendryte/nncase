// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Runtime.Interop;

/// <summary>
/// the Runtime Value.
/// </summary>
public abstract class RTValue : RTObject
{
    internal RTValue()
        : base(IntPtr.Zero)
    {
    }

    internal RTValue(IntPtr handle)
        : base(handle)
    {
    }

    /// <summary>
    /// convert IValue Value To RTValue.
    /// </summary>
    public static RTValue FromValue(IValue value) => value switch
    {
        TensorValue tv => RTTensor.FromTensor(tv.AsTensor()),
        TupleValue tv => RTTuple.FromTuple(tv),
        _ => throw new ArgumentOutOfRangeException(nameof(value)),
    };

    public static RTValue FromHandle(IntPtr handle, bool addRef = false)
    {
        try
        {
            Native.ValueIsTensor(handle, out var isTensor).ThrowIfFailed();
            return isTensor ? new RTTensor(handle, addRef) : new RTTuple(handle, addRef);
        }
        catch
        {
            Native.ObjectRelease(handle);
            throw;
        }
    }

    /// <summary>
    /// convert RT Value To IValue.
    /// </summary>
    public IValue ToValue() => this switch
    {
        RTTensor rTTensor => new TensorValue(rTTensor.ToTensor()),
        RTTuple rTTuple => new TupleValue(rTTuple.Fields.Select(f => f.ToValue()).ToArray()),
        _ => throw new ArgumentOutOfRangeException(),
    };
}

/// <summary>
/// Runtime tensor.
/// </summary>
public class RTTensor : RTValue
{
    private RTDataType? _elementType;
    private uint[]? _dimensions;
    private uint[]? _strides;

    internal RTTensor()
        : base(IntPtr.Zero)
    {
    }

    internal RTTensor(IntPtr handle, bool addRef = false)
        : base(handle)
    {
        if (addRef)
        {
            Native.ObjectAddRef(handle);
        }
    }

    /// <summary>
    /// Gets element type.
    /// </summary>
    public RTDataType ElementType
    {
        get
        {
            if (_elementType == null)
            {
                Native.TensorGetDtype(this, out var dtype).ThrowIfFailed();
                _elementType = dtype;
            }

            return _elementType;
        }
    }

    /// <summary>
    /// Gets get the buffer slice.
    /// </summary>
    public RTBufferSlice Buffer
    {
        get
        {
            Native.TensorGetBuffer(this, out var buffer).ThrowIfFailed();
            return RTBufferSlice.FromRT(buffer);
        }
    }

    /// <summary>
    /// Gets get the dimensions.
    /// </summary>
    public unsafe ReadOnlySpan<uint> Dimensions
    {
        get
        {
            if (_dimensions == null)
            {
                uint dimsLength = 0;
                Native.TensorGetDims(this, null, ref dimsLength);
                var dims = new uint[dimsLength];
                fixed (uint* dimsPtr = dims)
                {
                    Native.TensorGetDims(this, dimsPtr, ref dimsLength).ThrowIfFailed();
                }

                _dimensions = dims;
            }

            return _dimensions;
        }
    }

    /// <summary>
    /// Gets get the Strides.
    /// </summary>
    public unsafe ReadOnlySpan<uint> Strides
    {
        get
        {
            if (_strides == null)
            {
                uint stridesLength = 0;
                Native.TensorGetStrides(this, null, ref stridesLength);
                var strides = new uint[stridesLength];
                fixed (uint* stridesPtr = strides)
                {
                    Native.TensorGetStrides(this, stridesPtr, ref stridesLength).ThrowIfFailed();
                }

                _strides = strides;
            }

            return _strides;
        }
    }

    /// <inheritdoc/>
    public override bool IsInvalid => handle == IntPtr.Zero;

    /// <summary>
    /// Create runtime tensor.
    /// </summary>
    /// <param name="dataType">Data type.</param>
    /// <param name="dims">Dimensions.</param>
    /// <param name="strides">Strides.</param>
    /// <param name="bufferSlice">Buffer slice.</param>
    /// <returns>Created runtime tensor.</returns>
    public static unsafe RTTensor Create(RTDataType dataType, ReadOnlySpan<uint> dims, ReadOnlySpan<uint> strides, RTBufferSlice bufferSlice)
    {
        fixed (uint* dimsPtr = dims, stridesPtr = strides)
        {
            Native.TensorCreate(dataType, dimsPtr, (uint)dims.Length, stridesPtr, (uint)strides.Length, bufferSlice.ToRT(), out var tensor).ThrowIfFailed();
            return tensor;
        }
    }

    /// <summary>
    /// Create runtime tensor from tensor.
    /// </summary>
    /// <param name="tensor">Tensor.</param>
    /// <returns>Created runtime tensor.</returns>
    public static unsafe RTTensor FromTensor(Tensor tensor)
    {
        var dtype = (PrimType)tensor.ElementType;
        var sizeBytes = (uint)tensor.BytesBuffer.Length;
        var buffer = RTBufferAllocator.Host.Allocate(sizeBytes).AsHost()!;
        using (var mem = buffer.Map(RTMapAccess.Write))
        {
            tensor.BytesBuffer.CopyTo(mem.Memory.Span);
        }

        var dims = MemoryMarshal.Cast<int, uint>(tensor.Dimensions);
        var strides = MemoryMarshal.Cast<int, uint>(tensor.Strides);
        return Create(RTDataType.FromTypeCode(dtype.TypeCode), dims, strides, new RTBufferSlice { Buffer = buffer, Start = 0, SizeBytes = sizeBytes });
    }

    /// <summary>
    /// To <see cref="Tensor"/>.
    /// </summary>
    /// <returns>Copied tensor.</returns>
    public Tensor ToTensor()
    {
        var dtype = DataType.FromTypeCode(ElementType.TypeCode);
        var dims = MemoryMarshal.Cast<uint, int>(Dimensions);
        _ = MemoryMarshal.Cast<uint, int>(Strides);
        if (Buffer.SizeBytes > 0)
        {
            var hostBuffer = Buffer.Buffer.AsHost()!;
            using var owner = hostBuffer.Map(RTMapAccess.Read);
            return Tensor.FromBytes(new TensorType(dtype, new(dims.ToArray())), owner.Memory.ToArray());
        }

        return Tensor.FromBytes(new TensorType(dtype, new(dims.ToArray())), Array.Empty<byte>());
    }
}

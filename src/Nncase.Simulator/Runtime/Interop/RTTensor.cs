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

public abstract class RTValue : RTObject
{
    internal RTValue(IntPtr handle)
     : base(handle)
    {
    }

    internal static RTValue FromHandle(IntPtr handle)
    {
        try
        {
            Native.ValueIsTensor(handle, out var isTensor).ThrowIfFailed();
            return isTensor ? new RTTensor(handle) : new RTTuple(handle);
        }
        catch
        {
            Native.ObjectFree(handle);
            throw;
        }
    }

    /// <summary>
    /// convert RT Value To IValue
    /// </summary>
    /// <returns></returns>
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

    internal RTTensor(IntPtr handle)
        : base(handle)
    {
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
                Native.TensorGetDtype(Handle, out var dtype).ThrowIfFailed();
                _elementType = new RTDataType(dtype);
            }

            return _elementType;
        }
    }

    public RTBufferSlice Buffer
    {
        get
        {
            Native.TensorGetBuffer(Handle, out var buffer).ThrowIfFailed();
            return RTBufferSlice.FromRT(buffer);
        }
    }

    public unsafe ReadOnlySpan<uint> Dimensions
    {
        get
        {
            if (_dimensions == null)
            {
                uint dimsLength = 0;
                Native.TensorGetDims(Handle, null, ref dimsLength);
                var dims = new uint[dimsLength];
                fixed (uint* dimsPtr = dims)
                {
                    Native.TensorGetDims(Handle, dimsPtr, ref dimsLength).ThrowIfFailed();
                }

                _dimensions = dims;
            }

            return _dimensions;
        }
    }

    public unsafe ReadOnlySpan<uint> Strides
    {
        get
        {
            if (_strides == null)
            {
                uint stridesLength = 0;
                Native.TensorGetStrides(Handle, null, ref stridesLength);
                var strides = new uint[stridesLength];
                fixed (uint* stridesPtr = strides)
                {
                    Native.TensorGetStrides(Handle, stridesPtr, ref stridesLength).ThrowIfFailed();
                }

                _strides = strides;
            }

            return _strides;
        }
    }

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
            Native.TensorCreate(dataType.Handle, dimsPtr, (uint)dims.Length, stridesPtr, (uint)strides.Length, bufferSlice.ToRT(), out var tensor).ThrowIfFailed();
            return new RTTensor(tensor);
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
        var strides = MemoryMarshal.Cast<uint, int>(Strides);
        var hostBuffer = Buffer.Buffer.AsHost()!;
        using (var owner = hostBuffer.Map(RTMapAccess.Read))
        {
            return Tensor.FromBytes(new TensorType(dtype, new(dims)), owner.Memory.Span);
        }
    }
}

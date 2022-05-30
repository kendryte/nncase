// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime.Interop;

public abstract class RTValue : RTObject
{
    internal RTValue(IntPtr handle)
     : base(handle)
    {
    }
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
}

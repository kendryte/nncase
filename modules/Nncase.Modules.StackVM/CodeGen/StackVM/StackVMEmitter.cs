// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.CodeGen.StackVM;

/// <summary>
/// StackVM IL emitter.
/// </summary>
public sealed partial class StackVMEmitter : IDisposable
{
    private static readonly Dictionary<Type, byte> _typeCodes = new()
    {
        { typeof(sbyte), 0 },
        { typeof(short), 1 },
        { typeof(int), 2 },
        { typeof(long), 3 },
        { typeof(byte), 4 },
        { typeof(ushort), 5 },
        { typeof(uint), 6 },
        { typeof(ulong), 7 },
        { typeof(Half), 8 },
        { typeof(float), 9 },
        { typeof(double), 10 },
        { typeof(BFloat16), 11 },
    };

    /// <summary>
    /// Initializes a new instance of the <see cref="StackVMEmitter"/> class.
    /// </summary>
    /// <param name="writer">Code writer.</param>
    public StackVMEmitter(BinaryWriter writer)
    {
        _writer = writer;
        T = new TensorEmitter(this);
    }

    /// <summary>
    /// Gets tensor emitter.
    /// </summary>
    public TensorEmitter T { get; }

    /// <inheritdoc/>
    public void Dispose()
    {
        _writer.Dispose();
    }

    private void Write(byte value)
    {
        _writer.Write(value);
    }

    private void Write(ushort value)
    {
        _writer.Write(value);
    }

    private void Write(uint value)
    {
        _writer.Write(value);
    }

    private void Write(ulong value)
    {
        _writer.Write(value);
    }

    private void Write(sbyte value)
    {
        _writer.Write(value);
    }

    private void Write(short value)
    {
        _writer.Write(value);
    }

    private void Write(int value)
    {
        _writer.Write(value);
    }

    private void Write(long value)
    {
        _writer.Write(value);
    }

    private void Write(float value)
    {
        _writer.Write(value);
    }

    private void Write(double value)
    {
        _writer.Write(value);
    }

    private void Write(DataType value)
    {
        // TODO: Support generic datatype.
        switch (value)
        {
            case PrimType t:
                _writer.Write(ToTypeCode(t.CLRType));
                break;
            default:
                throw new ArgumentException($"Unsupported datatype: {value}");
        }
    }

    private byte ToTypeCode(Type clrType)
    {
        return _typeCodes[clrType];
    }

    /// <summary>
    /// Tensor emitter.
    /// </summary>
    public sealed partial class TensorEmitter
    {
        internal TensorEmitter(StackVMEmitter emitter)
        {
            _emitter = emitter;
        }
    }
}

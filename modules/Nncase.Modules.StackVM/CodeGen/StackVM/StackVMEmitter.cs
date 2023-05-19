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
public sealed partial class StackVMEmitter
{
    private static readonly Dictionary<Type, byte> _typeCodes = new()
    {
        { typeof(bool), 0 },
        { typeof(char), 1 },
        { typeof(sbyte), 2 },
        { typeof(short), 3 },
        { typeof(int), 4 },
        { typeof(long), 5 },
        { typeof(byte), 6 },
        { typeof(ushort), 7 },
        { typeof(uint), 8 },
        { typeof(ulong), 9 },
        { typeof(Half), 10 },
        { typeof(float), 11 },
        { typeof(double), 12 },
        { typeof(BFloat16), 13 },
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

    /// <summary>
    /// Gets position.
    /// </summary>
    public long Position => _writer.Position();

    /// <summary>
    /// write data type.
    /// </summary>
    public void Write(DataType value)
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

    private void Write(bool value)
    {
        _writer.Write(value);
    }

    private void Write(ReadOnlySpan<byte> value)
    {
        _writer.Write(value.Length);
        foreach (var b in value)
        {
            _writer.Write(b);
        }
    }

    private void Write(string value)
    {
        _writer.Write(Encoding.UTF8.GetBytes(value));
        _writer.Write((byte)0);
    }

    private void Write(string[] value)
    {
        foreach (var str in value)
        {
            Write(str);
        }

        _writer.Write((byte)0);
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
        public TensorEmitter(StackVMEmitter emitter)
        {
            _emitter = emitter;
        }
    }
}

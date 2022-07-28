// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase;

/// <summary>
/// Tensor or tuple of tensors.
/// </summary>
public interface IValue : IReadOnlyList<IValue>
{
    /// <summary>
    /// Gets type.
    /// </summary>
    IRType Type { get; }

    /// <summary>
    /// Get a single tensor.
    /// </summary>
    /// <returns>The single tensor.</returns>
    Tensor AsTensor();

    /// <summary>
    /// Get tensors.
    /// </summary>
    /// <returns>The tensors.</returns>
    Tensor[] AsTensors();
}

/// <summary>
/// Value extensions.
/// </summary>
public static class Value
{
    /// <summary>
    /// Create value form a tensor.
    /// </summary>
    /// <param name="tensor">The single tensor.</param>
    /// <returns>Created value.</returns>
    public static TensorValue FromTensor(Tensor tensor)
    {
        return new TensorValue(tensor);
    }

    /// <summary>
    /// Create value form tensors.
    /// </summary>
    /// <param name="tensors">The single tensor.</param>
    /// <returns>Created value.</returns>
    public static TupleValue FromTensors(params Tensor[] tensors)
    {
        return new TupleValue(tensors.Select(x => new TensorValue(x)).ToArray());
    }

    /// <summary>
    /// Create value form a constant.
    /// </summary>
    /// <param name="const">The constant.</param>
    /// <returns>Created value.</returns>
    public static IValue FromConst(Const @const)
    {
        if (@const is TensorConst tc)
        {
            return FromTensor(tc.Value);
        }
        else
        {
            var tpc = (TupleConst)@const;
            return new TupleValue(tpc.Fields.Select(x => FromConst(x)).ToArray());
        }
    }

    /// <summary>
    /// get the None Value.
    /// </summary>
    /// <returns></returns>
    public static IValue None => Nncase.NoneValue.Default;
}

/// <summary>
/// The None Value.
/// </summary>
public sealed class NoneValue : IValue
{
    /// <summary>
    /// Get the default None Value instane.
    /// </summary>
    public static readonly NoneValue Default = new();

    private NoneValue()
    {
    }

    /// <inheritdoc/>
    public IValue this[int index] => throw new InvalidOperationException("This Is None Value!");

    /// <inheritdoc/>
    public IRType Type => NoneType.Default;

    /// <inheritdoc/>
    public int Count => throw new InvalidOperationException("This Is None Value!");

    /// <inheritdoc/>
    public Tensor AsTensor()
    {
        throw new InvalidOperationException("This Is None Value!");
    }

    /// <inheritdoc/>
    public Tensor[] AsTensors()
    {
        throw new InvalidOperationException("This Is None Value!");
    }

    /// <inheritdoc/>
    public IEnumerator<IValue> GetEnumerator()
    {
        throw new InvalidOperationException("This Is None Value!");
    }
    /// <inheritdoc/>
    IEnumerator IEnumerable.GetEnumerator()
    {
        throw new InvalidOperationException("This Is None Value!");
    }
}

/// <summary>
/// Tensor value.
/// </summary>
public sealed class TensorValue : IValue, IEquatable<TensorValue?>
{
    private readonly Tensor _value;

    /// <summary>
    /// Initializes a new instance of the <see cref="TensorValue"/> class.
    /// </summary>
    /// <param name="tensor">Tensor.</param>
    public TensorValue(Tensor tensor)
    {
        _value = tensor;
        Type = new TensorType(_value.ElementType, _value.Shape);
    }

    /// <inheritdoc/>
    public int Count => 1;

    /// <inheritdoc/>
    public IRType Type { get; }

    /// <inheritdoc/>
    public IValue this[int index] => index == 0 ? this : throw new IndexOutOfRangeException();

    /// <inheritdoc/>
    public IEnumerator<IValue> GetEnumerator()
    {
        yield break;
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        yield break;
    }

    /// <inheritdoc/>
    public Tensor AsTensor()
    {
        return _value;
    }

    /// <inheritdoc/>
    public Tensor[] AsTensors()
    {
        return new[] { _value };
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        return Equals(obj as TensorValue);
    }

    /// <inheritdoc/>
    public bool Equals(TensorValue? other)
    {
        return other != null &&
               EqualityComparer<Tensor>.Default.Equals(_value, other._value);
    }

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        return HashCode.Combine(_value);
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        if (_value.BytesBuffer.Length <= 64)
            return _value.Shape.ToString() + " : " + _value.GetArrayString(false);
        return _value.Shape.ToString();
    }
}

/// <summary>
/// Tuple value.
/// </summary>
public sealed class TupleValue : IValue, IEquatable<TupleValue?>
{
    private readonly IValue[] _values;

    /// <summary>
    /// Initializes a new instance of the <see cref="TupleValue"/> class.
    /// </summary>
    /// <param name="values">Tuple fields.</param>
    public TupleValue(params IValue[] values)
    {
        _values = values;
        Type = new TupleType(values.Select(x => x.Type));
    }

    /// <inheritdoc/>
    public int Count => _values.Length;

    /// <inheritdoc/>
    public IRType Type { get; }

    /// <inheritdoc/>
    public IValue this[int index] => _values[index];

    /// <inheritdoc/>
    public Tensor AsTensor()
    {
        throw new InvalidOperationException();
    }

    /// <inheritdoc/>
    public IEnumerator<IValue> GetEnumerator()
    {
        return ((IEnumerable<IValue>)_values).GetEnumerator();
    }

    IEnumerator IEnumerable.GetEnumerator()
    {
        return GetEnumerator();
    }

    /// <inheritdoc/>
    public Tensor[] AsTensors()
    {
        return _values.Cast<TensorValue>().Select(x => x.AsTensor()).ToArray();
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        return Equals(obj as TupleValue);
    }

    /// <inheritdoc/>
    public bool Equals(TupleValue? other)
    {
        return other != null &&
               EqualityComparer<IValue[]>.Default.Equals(_values, other._values);
    }

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        return HashCode.Combine(_values);
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return "(" + string.Join(",", _values.Select(v => v.ToString())) + ")";
    }
}

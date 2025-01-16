// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CommunityToolkit.HighPerformance.Helpers;
using NetFabric.Hyperlinq;
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
    /// Gets get the None Value.
    /// </summary>
    public static IValue None => NoneValue.Default;

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
    /// Create value form a tensor.
    /// </summary>
    /// <param name="tensor">The single tensor.</param>
    /// <param name="ndSBP">NdSBP.</param>
    /// <param name="placement">Placement.</param>
    /// <returns>Created value.</returns>
    public static TensorValue FromTensor(Tensor tensor, IRArray<SBP> ndSBP, Placement placement)
    {
        return new TensorValue(tensor, ndSBP, placement);
    }

    public static TensorValue FromTensorLike(Tensor tensor, IRType type)
        => type switch
        {
            DistributedType dt => FromTensor(tensor, dt.NdSBP, dt.Placement),
            _ => FromTensor(tensor),
        };

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
        switch (@const)
        {
            case TensorConst tc:
                return FromTensor(tc.Value);
            case TupleConst tpc:
                return tpc.Value;
            case ShapeConst spc:
                return new ShapeValue(spc.Value.ToArray());
            case DimensionConst dc:
                return new DimensionValue(dc.Value);
            default:
                throw new ArgumentOutOfRangeException(nameof(@const));
        }
    }
}

/// <summary>
/// The None Value.
/// </summary>
public sealed class NoneValue : IValue, IEquatable<NoneValue?>
{
    /// <summary>
    /// Get the default None Value instane.
    /// </summary>
    public static readonly NoneValue Default = new();

    private NoneValue()
    {
    }

    /// <inheritdoc/>
    public IRType Type => NoneType.Default;

    /// <inheritdoc/>
    public int Count => 1;

    /// <inheritdoc/>
    public IValue this[int index] => index == 0 ? this : throw new ArgumentOutOfRangeException(nameof(index));

    public static bool operator ==(NoneValue? left, NoneValue? right) => true;

    public static bool operator !=(NoneValue? left, NoneValue? right) => false;

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
        yield break;
    }

    /// <inheritdoc/>
    IEnumerator IEnumerable.GetEnumerator()
    {
        yield break;
    }

    public override bool Equals(object? obj) => Equals(obj as NoneValue);

    public bool Equals(NoneValue? other) => other is not null;

    public override int GetHashCode() => 0;
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

    /// <summary>
    /// Initializes a new instance of the <see cref="TensorValue"/> class.
    /// </summary>
    /// <param name="tensor">Tensor.</param>
    /// <param name="ndSBP">NdSBP.</param>
    /// <param name="placement">Placement.</param>
    public TensorValue(Tensor tensor, IRArray<SBP> ndSBP, Placement placement)
    {
        _value = tensor;
        var tensorType = new TensorType(_value.ElementType, _value.Shape);
        Type = new DistributedType(tensorType, ndSBP, placement);
    }

    /// <inheritdoc/>
    public int Count => 1;

    /// <inheritdoc/>
    public IRType Type { get; }

    /// <inheritdoc/>
    public IValue this[int index] => index == 0 ? this : throw new ArgumentOutOfRangeException(nameof(index));

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
        {
            return _value.Shape.ToString() + " : " + _value.GetArrayString(false);
        }

        return _value.Shape.ToString();
    }
}

public sealed class DimensionValue : IValue, IEquatable<DimensionValue?>
{
    private readonly Dimension _value;

    public DimensionValue(Dimension value)
    {
        _value = value;
    }

    public IRType Type => new TensorType(DataTypes.Int64, Shape.Scalar);

    public int Count => 0;

    public Dimension Dimension => _value;

    public IValue this[int index] => throw new NotSupportedException("scalar can't index");

    public Tensor AsTensor() => throw new NotImplementedException();

    public Tensor[] AsTensors() => throw new NotImplementedException();

    public bool Equals(DimensionValue? other) => EqualityComparer<Dimension>.Default.Equals(_value, other?._value);

    public override int GetHashCode() => EqualityComparer<Dimension>.Default.GetHashCode(_value);

    public override bool Equals(object? obj)
    {
        return Equals(obj as DimensionValue);
    }

    public IEnumerator<IValue> GetEnumerator()
    {
        yield break;
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();
}

public sealed class ShapeValue : IValue, IEquatable<ShapeValue?>
{
    private readonly Dimension[] _values;

    public ShapeValue(params Dimension[] values)
    {
        _values = values;
    }

    public ShapeValue(IEnumerable<IValue> values)
    {
        var dims = new List<Dimension>();
        foreach (var item in values)
        {
            if (item is DimensionValue dimValue)
            {
                dims.Add(dimValue.Dimension);
            }
            else if (item is ShapeValue shapeValue)
            {
                dims.AddRange(shapeValue._values);
            }
            else
            {
                throw new NotSupportedException("only support dimension/shape value for constructor");
            }
        }

        _values = dims.ToArray();
    }

    public IRType Type => new TensorType(DataTypes.Int64, new[] { _values.Length });

    public int Count => _values.Length;

    public Span<Dimension> Dimensions => _values;

    public IValue this[int index] => new DimensionValue(_values[index]);

    public Tensor AsTensor() => throw new NotImplementedException();

    public Tensor[] AsTensors() => throw new NotImplementedException();

    public bool Equals(ShapeValue? other) => StructuralComparisons.StructuralEqualityComparer.Equals(_values, other?._values);

    public override int GetHashCode() => StructuralComparisons.StructuralEqualityComparer.GetHashCode(_values);

    public override bool Equals(object? obj)
    {
        return Equals(obj as ShapeValue);
    }

    public IEnumerator<IValue> GetEnumerator()
    {
        foreach (var item in _values)
        {
            yield return new DimensionValue(item);
        }
    }

    IEnumerator IEnumerable.GetEnumerator() => GetEnumerator();

    public override string ToString() => $"[{string.Join(",", _values.AsValueEnumerable().Select(v => v.ToString()))}]";
}

/// <summary>
/// Tuple value.
/// </summary>
public sealed class TupleValue : IValue, IEquatable<TupleValue?>
{
    public static readonly TupleValue Void = new TupleValue(ReadOnlySpan<IValue>.Empty);

    private readonly IValue[] _values;

    /// <summary>
    /// Initializes a new instance of the <see cref="TupleValue"/> class.
    /// </summary>
    /// <param name="values">Tuple fields.</param>
    public TupleValue(ReadOnlySpan<IValue> values)
    {
        _values = values.ToArray();
        Type = new TupleType(values.AsValueEnumerable().Select(x => x.Type).ToArray());
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
        return other != null && _values.SequenceEqual(other._values);
    }

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        return HashCode<IValue>.Combine(_values);
    }

    /// <inheritdoc/>
    public override string ToString()
    {
        return "(" + string.Join(",", _values.Select(v => v.ToString())) + ")";
    }
}

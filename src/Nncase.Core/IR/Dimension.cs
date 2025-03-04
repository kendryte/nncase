// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Passes;
using Nncase.Passes.Mutators;

namespace Nncase.IR;

/// <summary>
/// Dimension kind.
/// </summary>
public enum DimensionKind : byte
{
    /// <summary>
    /// Dynamic dimension.
    /// </summary>
    Dynamic,

    /// <summary>
    /// Fixed dimesnion.
    /// </summary>
    Fixed,

    /// <summary>
    /// Used for shape pattern.
    /// </summary>
    Unknown,
}

/// <summary>
/// Shape dimension.
/// </summary>
public struct Dimension : IEquatable<Dimension?>
{
    public static readonly Dimension Unknown = new Dimension(None.Default);

    private readonly long? _fixedValue;
    private readonly Expr? _exprValue;

    /// <summary>
    /// Initializes a new instance of the <see cref="Dimension"/> struct.
    /// </summary>
    /// <param name="value">Dimension value.</param>
    public Dimension(long value)
    {
        Kind = DimensionKind.Fixed;
        _fixedValue = value;
    }

    public Dimension(Expr value)
    {
        value = CompilerServices.FastSimplifyForDimension(value);
        if (value is TensorConst tc)
        {
            Kind = DimensionKind.Fixed;
            _fixedValue = tc.Value.ToScalar<long>();
        }
        else if (value is None)
        {
            Kind = DimensionKind.Unknown;
            _exprValue = None.Default;
        }
        else
        {
            Kind = DimensionKind.Dynamic;
            _exprValue = value;
        }
    }

    /// <summary>
    /// Gets kind.
    /// </summary>
    public DimensionKind Kind { get; }

    /// <summary>
    /// Gets value.
    /// </summary>
    public Expr Value => _exprValue ?? _fixedValue!.Value;

    /// <summary>
    /// Gets FixedValue.
    /// </summary>
    public long FixedValue
    {
        get => _fixedValue ??
           throw new InvalidOperationException("Only Can Get It When Shape Is Fixed !");
    }

    /// <summary>
    /// Gets a value indicating whether dynamic.
    /// </summary>
    public bool IsDynamic => Kind is DimensionKind.Dynamic;

    /// <summary>
    /// Gets a value indicating whether fixed.
    /// </summary>
    public bool IsFixed => Kind == DimensionKind.Fixed;

    public bool IsUnknown => Kind == DimensionKind.Unknown;

    /// <summary>
    /// Convert <see cref="long"/> to a fixed <see cref="Dimension"/>.
    /// </summary>
    /// <param name="value">Dimension value.</param>
    public static implicit operator Dimension(long value) => new(value);

    /// <summary>
    /// Convert <see cref="Expr"/> to a <see cref="Dimension"/> expression.
    /// </summary>
    /// <param name="value">Dimension value.</param>
    public static implicit operator Dimension(Expr value) => value switch
    {
        TensorConst dc => new(dc.Value.ToScalar<long>()),
        _ => new(value),
    };

    public static bool operator ==(Dimension left, Dimension right)
    {
        return left.Equals(right);
    }

    public static bool operator !=(Dimension left, Dimension right)
    {
        return !(left == right);
    }

    public static Dimension operator +(Dimension lhs, Dimension rhs) => (lhs.IsFixed, rhs.IsFixed) switch
    {
        (true, true) => lhs.FixedValue + rhs.FixedValue,
        (true, _) when lhs.FixedValue == 0 => rhs,
        (_, true) when rhs.FixedValue == 0 => lhs,
        (_, _) when lhs.IsUnknown || rhs.IsUnknown => Unknown,
        (_, _) => new Dimension(lhs.Value + rhs.Value),
    };

    public static Dimension operator +(Dimension lhs, int rhs) => lhs.IsFixed ? lhs.FixedValue + rhs : new Dimension(lhs.Value + rhs);

    public static Dimension operator -(Dimension lhs, Dimension rhs) => (lhs.IsFixed, rhs.IsFixed) switch
    {
        (true, true) => lhs.FixedValue - rhs.FixedValue,
        (_, true) when rhs.FixedValue == 0 => lhs,
        (_, _) when lhs.IsUnknown || rhs.IsUnknown => Unknown,
        (_, _) => new Dimension(lhs.Value - rhs.Value),
    };

    public static Dimension operator *(Dimension lhs, Dimension rhs) => (lhs.IsFixed, rhs.IsFixed) switch
    {
        (true, true) => lhs.FixedValue * rhs.FixedValue,
        (true, _) when lhs.FixedValue == 1 => rhs,
        (_, true) when rhs.FixedValue == 1 => lhs,
        (_, _) when lhs.IsUnknown || rhs.IsUnknown => Unknown,
        (_, _) => new Dimension(lhs.Value * rhs.Value),
    };

    public static Dimension operator /(Dimension lhs, Dimension rhs) => (lhs.IsFixed, rhs.IsFixed) switch
    {
        (true, true) => lhs.FixedValue / rhs.FixedValue,
        (_, _) when lhs.IsUnknown || rhs.IsUnknown => Unknown,
        (_, _) => new Dimension(lhs.Value / rhs.Value),
    };

    public static Dimension operator %(Dimension lhs, Dimension rhs) => (lhs.IsFixed, rhs.IsFixed) switch
    {
        (true, true) => lhs.FixedValue % rhs.FixedValue,
        (_, _) when lhs.IsUnknown || rhs.IsUnknown => Unknown,
        (_, _) => new Dimension(lhs.Value % rhs.Value),
    };

    public static Dimension Abs(Dimension value)
    {
        if (value.IsFixed)
        {
            return System.Math.Abs(value.FixedValue);
        }

        return value.Value.Metadata.Range?.Min >= 0 ? value.Value : IR.F.Math.Abs(value.Value);
    }

    public static Dimension Clamp(Dimension value, Dimension min, Dimension max)
    {
        if (value.IsFixed && min.IsFixed && max.IsFixed)
        {
            return System.Math.Clamp(value.FixedValue, min.FixedValue, max.FixedValue);
        }

        return IR.F.Math.Clamp(value.Value, min.Value, max.Value);
    }

    public static Dimension CeilDiv(Dimension lhs, Dimension rhs)
    {
        if (lhs.IsFixed && rhs.IsFixed)
        {
            return (lhs.FixedValue + rhs.FixedValue - 1) / rhs.FixedValue;
        }

        return IR.F.Math.CeilDiv(lhs.Value, rhs.Value);
    }

    public static Dimension Max(Dimension lhs, Dimension rhs)
    {
        if (lhs.IsFixed && rhs.IsFixed)
        {
            return System.Math.Max(lhs.FixedValue, rhs.FixedValue);
        }

        return IR.F.Math.Max(lhs.Value, rhs.Value);
    }

    public static Dimension Min(Dimension lhs, Dimension rhs)
    {
        if (lhs.IsFixed && rhs.IsFixed)
        {
            return System.Math.Min(lhs.FixedValue, rhs.FixedValue);
        }

        return IR.F.Math.Min(lhs.Value, rhs.Value);
    }

    /// <inheritdoc/>
    public override string ToString() => Kind switch
    {
        DimensionKind.Dynamic when Value is Var var => $"%{var.Name}",
        DimensionKind.Fixed => FixedValue.ToString(),
        DimensionKind.Unknown => "?",
        _ => "...",
    };

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        return obj is Dimension dimension && Equals(dimension);
    }

    /// <inheritdoc/>
    public bool Equals(Dimension? other)
    {
        return other is not null && (Kind, other.Value.Kind) switch
        {
            (DimensionKind.Dynamic, DimensionKind.Dynamic) => Value == other.Value.Value,
            (DimensionKind.Fixed, DimensionKind.Fixed) => FixedValue == other.Value.FixedValue,
            (DimensionKind.Unknown, DimensionKind.Unknown) => true,
            (_, _) => false,
        };
    }

    /// <inheritdoc/>
    public override int GetHashCode()
    {
        return IsFixed ? HashCode.Combine(Kind, FixedValue) : HashCode.Combine(Kind, Value);
    }

    public bool HasFixedValue(Predicate<long> predicate)
    {
        return IsFixed && predicate(FixedValue);
    }

    public bool IsAssignableFrom(Dimension dimension) =>
        (Kind, dimension.Kind) switch
        {
            (DimensionKind.Dynamic, DimensionKind.Dynamic) => Value == dimension.Value,
            (DimensionKind.Dynamic, DimensionKind.Fixed) => Value.Metadata.Range?.Contains(dimension.FixedValue) ?? true,
            (DimensionKind.Fixed, DimensionKind.Fixed) => FixedValue == dimension.FixedValue,
            (DimensionKind.Unknown, _) => true,
            (_, _) => false,
        };

    public Expr ToExpr() => IsFixed ? FixedValue : Value;
}

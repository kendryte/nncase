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
    /// Fixed dimesnion.
    /// </summary>
    Fixed,

    /// <summary>
    /// Dynamic dimension.
    /// </summary>
    Dynamic,

    /// <summary>
    /// Used for shape pattern.
    /// </summary>
    Unknown,
}

/// <summary>
/// Shape dimension.
/// </summary>
public sealed class Dimension : Expr, IEquatable<Dimension?>
{
    public static readonly Dimension Unknown = new Dimension(None.Default);

    public Dimension(DimExpr value)
        : base([value])
    {
        if (value is dim tc)
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
    public DimExpr Value => (DimExpr)Operands[0];

    /// <summary>
    /// Gets FixedValue.
    /// </summary>
    public long FixedValue => IsFixed ? _fixedValue : throw new InvalidOperationException("Dimension is not fixed.");

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
        else if (value.IsUnknown)
        {
            return Unknown;
        }

        return value.Value.Metadata.Range?.Min >= 0 ? value.Value : IR.F.Math.Abs(value.Value);
    }

    public static Dimension Clamp(Dimension value, Dimension min, Dimension max)
    {
        if (value.IsFixed && min.IsFixed && max.IsFixed)
        {
            return System.Math.Clamp(value.FixedValue, min.FixedValue, max.FixedValue);
        }
        else if (value.IsFixed && min.IsFixed && value.FixedValue <= min.FixedValue)
        {
            return min;
        }
        else if (value.IsFixed && max.IsFixed && value.FixedValue >= max.FixedValue)
        {
            return max;
        }
        else if (value.IsUnknown || min.IsUnknown || max.IsUnknown)
        {
            return Unknown;
        }

        return IR.F.Math.Clamp(value.Value, min.Value, max.Value);
    }

    public static Dimension CeilDiv(Dimension lhs, Dimension rhs)
    {
        if (lhs.IsFixed && rhs.IsFixed)
        {
            return (lhs.FixedValue + rhs.FixedValue - 1) / rhs.FixedValue;
        }
        else if (lhs.IsUnknown || rhs.IsUnknown)
        {
            return Unknown;
        }

        return IR.F.Math.CeilDiv(lhs.Value, rhs.Value);
    }

    public static Dimension AlignUp(Dimension dimension, int align)
    {
        return CeilDiv(dimension, align) * align;
    }

    public static Dimension Max(Dimension lhs, Dimension rhs)
    {
        if (lhs.IsFixed && rhs.IsFixed)
        {
            return System.Math.Max(lhs.FixedValue, rhs.FixedValue);
        }
        else if (lhs.IsUnknown || rhs.IsUnknown)
        {
            return Unknown;
        }

        return IR.F.Math.Max(lhs.Value, rhs.Value);
    }

    public static Dimension Min(Dimension lhs, Dimension rhs)
    {
        if (lhs.IsFixed && rhs.IsFixed)
        {
            return System.Math.Min(lhs.FixedValue, rhs.FixedValue);
        }
        else if (lhs.IsUnknown || rhs.IsUnknown)
        {
            return Unknown;
        }

        return IR.F.Math.Min(lhs.Value, rhs.Value);
    }

    public static Dimension Select(Dimension value, Dimension compare, Dimension trueValue, Dimension falseValue)
    {
        if (trueValue == falseValue)
        {
            return trueValue;
        }
        else if (value.IsFixed && compare.IsFixed)
        {
            return value.FixedValue == compare.FixedValue ? trueValue : falseValue;
        }
        else if (value.Value.Metadata?.Range is { Min: var min, Max: var max }
                && compare.IsFixed
                && (min > compare.FixedValue || max < compare.FixedValue))
        {
            return falseValue;
        }
        else if (value.IsUnknown || compare.IsUnknown)
        {
            return Unknown;
        }

        return IR.F.Math.Select(IR.F.Math.Equal(value.Value, compare.Value), trueValue.ToExpr(), falseValue.ToExpr());
    }

    public static Expr ConcatPadding(Dimension[] padH, Dimension[] padW)
    {
        // return [[padh_before, padh_after],
        //         [padw_before, padw_after]]
        var padHExpr = new Shape(padH).ToValueArrayExpr();
        var padWExpr = new Shape(padW).ToValueArrayExpr();
        var result = IR.F.Tensors.Stack(new IR.Tuple(padHExpr, padWExpr), 0);
        return padHExpr is Const && padWExpr is Const ? result.Evaluate().AsTensor() : result;
    }

    public static Expr ConcatPadding(Dimension[,] pads)
    {
        if (pads.GetLength(1) != 2)
        {
            throw new ArgumentException("Padding must be a 2D array with 2 columns");
        }

        var stackedPads = Enumerable.Range(0, pads.GetLength(0))
            .Select(i => new Shape(pads[i, 0], pads[i, 1]).ToValueArrayExpr())
            .ToArray();
        var result = IR.F.Tensors.Stack(new IR.Tuple(stackedPads), 0);
        return stackedPads.All(x => x is Const) ? result.Evaluate().AsTensor() : result;
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

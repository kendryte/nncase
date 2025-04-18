// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR.Shapes;
using Nncase.IR.Tensors;
using Nncase.Passes;
using Nncase.Passes.Mutators;
using Nncase.Utilities;

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

public static class DimensionExtensions
{
    public static Dimension AsDim(this Expr expr) => expr switch
    {
        TensorConst tc => tc.Value.ToScalar<long>(),
        Dimension dim => dim,
        _ => new AsDim(expr),
    };

    public static Shape AsShape(this Expr value)
    {
        if (value is TensorConst tc)
        {
            return new Shape(tc.Value.ToArray<long>());
        }
        else if (value is Shape shape)
        {
            return shape;
        }
        else
        {
            shape = value.CheckedShape;
            if (shape.Rank != 1 || !shape.IsFixed)
            {
                return Shape.Unranked;
            }

            var rank = (int)shape[0].FixedValue;
            return new Shape(Enumerable.Range(0, rank).Select(x => value[x].AsDim()));
        }
    }
}

/// <summary>
/// Shape dimension.
/// </summary>
public abstract class Dimension : Expr
{
    public static readonly DimConst Zero = new(0);
    public static readonly DimConst One = new(1);
    public static readonly DimConst MinusOne = new(-1);
    public static readonly Dimension Unknown = UnknownDim.Default;

    /// <summary>
    /// Initializes a new instance of the <see cref="Dimension"/> class.
    /// </summary>
    /// <param name="operands">Operands.</param>
    protected Dimension(Expr[] operands)
        : base(operands)
    {
    }

    /// <summary>
    /// Gets kind.
    /// </summary>
    public abstract DimensionKind Kind { get; }

    /// <summary>
    /// Gets FixedValue.
    /// </summary>
    public virtual long FixedValue => throw new InvalidOperationException("Dimension is not fixed.");

    /// <summary>
    /// Gets a value indicating whether dynamic.
    /// </summary>
    public bool IsDynamic => Kind is DimensionKind.Dynamic;

    /// <summary>
    /// Gets a value indicating whether fixed.
    /// </summary>
    public bool IsFixed => Kind == DimensionKind.Fixed;

    public bool IsUnknown => Kind == DimensionKind.Unknown;

    public static implicit operator Dimension(string name) => new DimVar(name);

    /// <summary>
    /// Convert <see cref="long"/> to a fixed <see cref="Dimension"/>.
    /// </summary>
    /// <param name="value">Dimension value.</param>
    public static implicit operator Dimension(long value) => new DimConst(value);

    /// <summary>
    /// Convert <see cref="int"/> to a fixed <see cref="Dimension"/>.
    /// </summary>
    /// <param name="value">Dimension value.</param>
    public static implicit operator Dimension(int value) => new DimConst(value);

    public static Dimension operator -(Dimension value) => value * -1;

    public static Dimension operator +(Dimension lhs, Dimension rhs) => (lhs, rhs) switch
    {
        (DimConst lhsConst, DimConst rhsConst) => lhsConst.Value + rhsConst.Value,
        (DimConst dimConst, _) when dimConst.Value == 0 => rhs,
        (_, DimConst dimConst) when dimConst.Value == 0 => lhs,
        (_, _) when lhs.IsUnknown || rhs.IsUnknown => Unknown,
        (DimSum lhsSum, DimSum rhsSum) => new DimSum(SpanUtility.Concat(lhsSum.Operands, rhsSum.Operands)).Simplify(),
        (DimSum lhsSum, _) => new DimSum(SpanUtility.Concat(lhsSum.Operands, [rhs])).Simplify(),
        (_, DimSum rhsSum) => new DimSum(SpanUtility.Concat([lhs], rhsSum.Operands)).Simplify(),
        (_, _) => new DimSum([lhs, rhs]).Simplify(),
    };

    public static Dimension operator -(Dimension lhs, Dimension rhs) => lhs + (-rhs);

    public static Dimension operator *(Dimension lhs, Dimension rhs) => (lhs, rhs) switch
    {
        (DimConst lhsConst, DimConst rhsConst) => lhsConst.Value * rhsConst.Value,
        (DimConst lhsConst, _) when lhsConst.Value == 0 => 0,
        (_, DimConst rhsConst) when rhsConst.Value == 0 => 0,
        (DimConst dimConst, _) when dimConst.Value == 1 => rhs,
        (_, DimConst dimConst) when dimConst.Value == 1 => lhs,
        (_, _) when lhs.IsUnknown || rhs.IsUnknown => Unknown,
        (DimSum lhsSum, _) => new DimSum(lhsSum.Operands.AsValueEnumerable().Select(x => x * rhs).ToArray()).Simplify(),
        (_, DimSum rhsSum) => new DimSum(rhsSum.Operands.AsValueEnumerable().Select(x => lhs * x).ToArray()).Simplify(),
        (DimProduct dimProduct, DimConst dimConst) => dimProduct.With(scale: dimProduct.Scale * dimConst.Value),
        (DimConst dimConst, DimProduct dimProduct) => dimProduct.With(scale: dimProduct.Scale * dimConst.Value),
        (DimProduct lhsProduct, DimProduct rhsProduct) => new DimProduct(SpanUtility.Concat(lhsProduct.Operands, rhsProduct.Operands)).Simplify(),
        (DimProduct lhsProduct, _) => new DimProduct(SpanUtility.Concat(lhsProduct.Operands, [rhs])).Simplify(),
        (_, DimProduct rhsProduct) => new DimProduct(SpanUtility.Concat([lhs], rhsProduct.Operands)).Simplify(),
        (_, _) => new DimProduct([lhs, rhs]).Simplify(),
    };

    public static Dimension operator /(Dimension lhs, Dimension rhs) => (lhs, rhs) switch
    {
        (DimConst lhsConst, DimConst rhsConst) => lhsConst.Value / rhsConst.Value,
        (_, DimConst dimConst) when dimConst.Value == 0 => throw new DivideByZeroException(),
        (_, DimConst dimConst) when dimConst.Value == 1 => lhs,
        (_, _) when lhs.IsUnknown || rhs.IsUnknown => Unknown,
        (DimProduct dimProduct, DimConst dimConst) when dimProduct.Scale % dimConst.Value == 0 => dimProduct.With(scale: dimProduct.Scale / dimConst.Value),
        (_, _) => new DimFraction(DimDivideMode.FloorDiv, lhs, rhs).Simplify(),
    };

    public static Dimension operator %(Dimension lhs, Dimension rhs) => (lhs, rhs) switch
    {
        (DimConst lhsConst, DimConst rhsConst) => lhsConst.Value % rhsConst.Value,
        (_, DimConst dimConst) when dimConst.Value == 0 => throw new DivideByZeroException(),
        (_, DimConst dimConst) when dimConst.Value == 1 => DimConst.Zero,
        (_, _) when lhs.IsUnknown || rhs.IsUnknown => Unknown,
        (DimProduct dimProduct, DimConst dimConst) when dimProduct.Scale % dimConst.Value == 0 => DimConst.Zero,
        (_, _) => new DimRemainder(lhs, rhs).Simplify(),
    };

    public static Dimension Abs(Dimension value) => value switch
    {
        _ when value.Metadata.Range?.Min >= 0 => value,
        DimConst dimConst => System.Math.Abs(dimConst.Value),
        UnknownDim => Unknown,
        DimProduct dimProduct => dimProduct.With(operands: dimProduct.Operands.AsValueEnumerable().Select(Abs).ToArray(), scale: System.Math.Abs(dimProduct.Scale)),
        DimFraction dimFraction => dimFraction.With(numerator: Abs(dimFraction.Numerator), denominator: Abs(dimFraction.Denominator)),
        DimRemainder dimRemainder => new DimRemainder(Abs(dimRemainder.Numerator), Abs(dimRemainder.Denominator)),
        DimSum dimSum => dimSum.With(dimSum.Operands.AsValueEnumerable().Select(Abs).ToArray(), bias: System.Math.Abs(dimSum.Bias)).Simplify(),
        _ => new DimAbs(value),
    };

    public static Dimension Pow(Dimension value, int power)
    {
        return value switch
        {
            DimConst dimConst => (long)System.Math.Pow(dimConst.Value, power),
            OpaqueDim opaqueDim => new DimPower(opaqueDim, power),
            UnknownDim => UnknownDim.Default,
            DimProduct dimProduct => dimProduct.With(operands: dimProduct.Operands.AsValueEnumerable().Select(x => Pow(x, power)).ToArray(), scale: (long)System.Math.Pow(dimProduct.Scale, power)),
            DimAbs dimAbs => power % 2 == 0 ? dimAbs.Operand : new DimAbs(Pow(dimAbs.Operand, power)),
            _ => throw new NotSupportedException($"Unsupported dimension type: {value.GetType()}"),
        };
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

        return new DimClamp(value, min, max);
    }

    public static Dimension CeilDiv(Dimension lhs, Dimension rhs) => (lhs, rhs) switch
    {
        (DimConst lhsConst, DimConst rhsConst) => lhsConst.Value / rhsConst.Value,
        (_, DimConst dimConst) when dimConst.Value == 0 => throw new DivideByZeroException(),
        (_, DimConst dimConst) when dimConst.Value == 1 => lhs,
        (_, _) when lhs.IsUnknown || rhs.IsUnknown => Unknown,
        (DimProduct dimProduct, DimConst dimConst) when dimProduct.Scale % dimConst.Value == 0 => dimProduct.With(scale: dimProduct.Scale / dimConst.Value),
        (_, _) => new DimFraction(DimDivideMode.CeilDiv, lhs, rhs).Simplify(),
    };

    public static Dimension AlignUp(Dimension dimension, int align)
    {
        return CeilDiv(dimension, align) * align;
    }

    public static Dimension Max(params Dimension[] dimensions)
    {
        if (dimensions.Length == 0)
        {
            throw new ArgumentException("At least one dimension is required.");
        }

        if (dimensions.All(x => x.IsFixed))
        {
            return dimensions.MaxBy(x => x.FixedValue)!;
        }
        else if (dimensions.Any(x => x.IsUnknown))
        {
            return Unknown;
        }

        return new DimMax(dimensions).Simplify();
    }

    public static Dimension Min(params Dimension[] dimensions)
    {
        if (dimensions.Length == 0)
        {
            throw new ArgumentException("At least one dimension is required.");
        }

        if (dimensions.All(x => x.IsFixed))
        {
            return dimensions.MinBy(x => x.FixedValue)!;
        }
        else if (dimensions.Any(x => x.IsUnknown))
        {
            return Unknown;
        }

        return new DimMin(dimensions).Simplify();
    }

    public static Dimension Select(Dimension value, Dimension expected, Dimension trueValue, Dimension falseValue)
    {
        if (trueValue == falseValue)
        {
            return trueValue;
        }
        else if (value.IsFixed && expected.IsFixed)
        {
            return value.FixedValue == expected.FixedValue ? trueValue : falseValue;
        }
        else if (value.Metadata?.Range is { Min: var min, Max: var max }
                && expected.IsFixed
                && (min > expected.FixedValue || max < expected.FixedValue))
        {
            return falseValue;
        }
        else if (value.IsUnknown || expected.IsUnknown)
        {
            return Unknown;
        }

        return new DimCompareAndSelect(value, expected, trueValue, falseValue);
    }

    public static Dimension Positive(Dimension value, Dimension extent)
    {
        if (value.IsFixed)
        {
            return value.FixedValue >= 0 ? value : value + extent;
        }
        else if (value.IsUnknown || extent.IsUnknown)
        {
            return Unknown;
        }
        else if (value.Metadata.Range?.Min >= 0)
        {
            return value;
        }

        return new DimPositive(value, extent);
    }

    public static bool TryDivExactly(Dimension numerator, Dimension denominator, [MaybeNullWhen(false)] out Dimension divided)
    {
        var remainder = numerator % denominator;
        divided = remainder switch
        {
            DimConst dimConst => dimConst.Value == 0 ? numerator / denominator : null,
            _ => numerator / denominator,
        };
        return divided != null;
    }

    public static Paddings ConcatPadding(Dimension[] padH, Dimension[] padW)
    {
        // return [[padh_before, padh_after],
        //         [padw_before, padw_after]]
        return new Paddings(new Padding(padH[0], padH[1]), new Padding(padW[0], padW[1]));
    }

    public static Paddings ConcatPadding(Dimension[,] pads)
    {
        if (pads.GetLength(1) != 2)
        {
            throw new ArgumentException("Padding must be a 2D array with 2 columns");
        }

        var paddings = new Padding[pads.GetLength(0)];
        for (int i = 0; i < pads.GetLength(0); i++)
        {
            paddings[i] = new Padding(pads[i, 0], pads[i, 1]);
        }

        return new Paddings(paddings);
    }

    public bool HasFixedValue(Predicate<long> predicate)
    {
        return IsFixed && predicate(FixedValue);
    }

    public bool IsAssignableFrom(Dimension dimension) =>
        (Kind, dimension.Kind) switch
        {
            (DimensionKind.Dynamic, DimensionKind.Dynamic) => this == dimension,
            (DimensionKind.Dynamic, DimensionKind.Fixed) => Metadata.Range?.Contains(dimension.FixedValue) ?? true,
            (DimensionKind.Fixed, DimensionKind.Fixed) => FixedValue == dimension.FixedValue,
            (DimensionKind.Unknown, _) => true,
            (_, _) => false,
        };

    public virtual Dimension Simplify() => this;
}

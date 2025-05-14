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
    public static Dimension AsDim(this BaseExpr expr) => expr switch
    {
        TensorConst tc => tc.Value.ToScalar<long>(),
        Dimension dim => dim,
        Expr e => new AsDim(e),
        _ => throw new ArgumentException($"Cannot convert {expr} to dimension."),
    };

    public static Shape AsShape(this BaseExpr value)
    {
        if (value is TensorConst tc)
        {
            return new RankedShape(tc.Value.ToArray<long>());
        }
        else if (value is Shape shapeExpr)
        {
            return shapeExpr;
        }
        else if (value is Call { Target: Concat } concat)
        {
            if (concat[Concat.Input] is Tuple tuple)
            {
                return new RankedShape(tuple.Fields.AsValueEnumerable().Select(x => GetItem(x, 0).AsDim()).ToArray());
            }
        }
        else if (value is Call { Target: Stack } stack)
        {
            if (stack[Stack.Inputs] is Tuple tuple)
            {
                return new RankedShape(tuple.Fields.AsValueEnumerable().Select(x => x.AsDim()).ToArray());
            }
        }

        var shape = value.CheckedShape;
        if (shape.Rank != 1)
        {
            return Shape.Invalid;
        }
        else if (!shape.IsFixed)
        {
            return new UnrankedShape((Expr)value);
        }

        var rank = (int)shape[0].FixedValue;
        return new RankedShape(Enumerable.Range(0, rank).Select(x => GetItem(value, x).AsDim()));
    }

    public static Padding AsPadding(this BaseExpr value)
    {
        if (value is TensorConst tc)
        {
            return tc.Value.Cast<long>();
        }
        else if (value is Padding padding)
        {
            return padding;
        }
        else if (value is Call { Target: Concat } concat)
        {
            if (concat[Concat.Input] is Tuple tuple)
            {
                return tuple.Fields.AsValueEnumerable().Select(x => GetItem(x, 0).AsDim()).ToArray();
            }
        }
        else if (value is Call { Target: Stack } stack)
        {
            if (stack[Stack.Inputs] is Tuple tuple)
            {
                return tuple.Fields.AsValueEnumerable().Select(x => x.AsDim()).ToArray();
            }
        }

        throw new ArgumentException($"Cannot convert {value} to padding.");
    }

    public static Paddings AsPaddings(this BaseExpr value)
    {
        if (value is TensorConst tc)
        {
            return new Paddings(tc.Value.Cast<long>());
        }
        else if (value is Paddings paddings)
        {
            return paddings;
        }
        else if (value is Call { Target: Concat } concat)
        {
            if (concat[Concat.Input] is Tuple tuple)
            {
                return new Paddings(tuple.Fields.AsValueEnumerable().Select(x => x.AsPadding()).ToArray());
            }
        }
        else if (value is Call { Target: Stack } stack)
        {
            if (stack[Stack.Inputs] is Tuple tuple)
            {
                return new Paddings(tuple.Fields.AsValueEnumerable().Select(x => x.AsPadding()).ToArray());
            }
        }

        throw new ArgumentException($"Cannot convert {value} to paddings.");
    }

    private static BaseExpr GetItem(BaseExpr expr, int index) => expr switch
    {
        Expr e => e[index],
        Shape s => s[index],
        _ => throw new ArgumentException($"Cannot get item from {expr}"),
    };
}

/// <summary>
/// Shape dimension.
/// </summary>
public abstract class Dimension : BaseExpr
{
    public static readonly DimConst Zero = new(0);
    public static readonly DimConst One = new(1);
    public static readonly DimConst MinusOne = new(-1);
    public static readonly Dimension Unknown = UnknownDim.Default;

    /// <summary>
    /// Initializes a new instance of the <see cref="Dimension"/> class.
    /// </summary>
    /// <param name="operands">Operands.</param>
    protected Dimension(IEnumerable<BaseExpr> operands)
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

    public override BaseExpr this[Dimension index] => throw new InvalidOperationException();

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

    public static bool operator ==(Dimension? left, Dimension? right) => EqualityComparer<Dimension>.Default.Equals(left, right);

    public static bool operator !=(Dimension? left, Dimension? right) => !(left == right);

    public static Dimension operator -(Dimension value) => value * -1;

    public static Dimension operator +(Dimension lhs, Dimension rhs) => DimSum.TrySimplify(0, [lhs, rhs]) ?? new DimSum([lhs, rhs]);

    public static Dimension operator -(Dimension lhs, Dimension rhs) => lhs + (-rhs);

    public static Dimension operator *(Dimension lhs, Dimension rhs) => DimProduct.TrySimplify(1, [lhs, rhs]) ?? new DimProduct([lhs, rhs]);

    public static Dimension operator /(Dimension lhs, Dimension rhs) => DimFraction.Simplify(DimDivideMode.FloorDiv, lhs, rhs);

    public static Dimension operator %(Dimension lhs, Dimension rhs) => DimRemainder.Simplify(lhs, rhs);

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
            _ when power == 0 => One,
            _ when power == 1 => value,
            _ when power == -1 => One / value,
            DimConst dimConst => (long)System.Math.Pow(dimConst.Value, power),
            OpaqueDim opaqueDim => new DimPower(opaqueDim, power),
            UnknownDim => UnknownDim.Default,
            DimProduct dimProduct => dimProduct.With(operands: dimProduct.Operands.AsValueEnumerable().Select(x => Pow(x, power)).ToArray(), scale: (long)System.Math.Pow(dimProduct.Scale, power)),
            DimAbs dimAbs => power % 2 == 0 ? Pow(dimAbs.Operand, power) : new DimAbs(Pow(dimAbs.Operand, power)),
            DimSum dimSum => DimHelpers.Pow(dimSum, power),
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

    public static Dimension CeilDiv(Dimension lhs, Dimension rhs) => DimFraction.Simplify(DimDivideMode.CeilDiv, lhs, rhs);

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

    public static Dimension Min(params Dimension[] dimensions) => DimMin.TrySimplify(dimensions) ?? new DimMin(dimensions);

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

    public override bool Equals(object? obj) => base.Equals(obj as Dimension);

    protected override void OnOperandsReplaced()
    {
        base.OnOperandsReplaced();

        if (Kind == DimensionKind.Dynamic
            && Operands.AsValueEnumerable().Any(x => x is DimConst or Const or RankedShape { IsFixed: true }))
        {
            ReplaceAllUsesWith(Simplify());
        }
    }
}

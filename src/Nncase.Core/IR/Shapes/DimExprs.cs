// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using NetFabric.Hyperlinq;
using Nncase.IR.Shapes;
using Nncase.Utilities;

namespace Nncase.IR;

public sealed class DimProduct : Dimension, IEquatable<DimProduct?>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DimProduct"/> class.
    /// </summary>
    /// <param name="operands">Operands.</param>
    /// <param name="scale">Scale.</param>
    public DimProduct(Dimension[] operands, long scale = 1)
        : base(operands)
    {
        Scale = scale;
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    public long Scale { get; }

    /// <summary>
    /// Gets operands.
    /// </summary>
    public new ReadOnlySpan<Dimension> Operands => SpanUtility.UnsafeCast<BaseExpr, Dimension>(base.Operands);

    public int Count => Operands.Length;

    public Dimension this[int index] => Operands[index];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimProduct(this, context);

    public Dimension With(Dimension[]? operands = null, long? scale = null)
    {
        if (operands == null)
        {
            return new DimProduct(Operands.ToArray(), scale ?? Scale);
        }

        return TrySimplify(scale ?? Scale, operands) ?? new DimProduct(operands, scale ?? Scale);
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimProduct);

    /// <inheritdoc/>
    public bool Equals(DimProduct? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Scale == other.Scale && Operands.SequenceEqual(other.Operands);
    }

    public override string ToString()
    {
        var scale = Scale == 1 ? string.Empty : $"{Scale} * ";
        return $"({scale}{StringUtility.Join(" * ", Operands)})";
    }

    public override Dimension Simplify() => TrySimplify(Scale, Operands) ?? this;

    internal static Dimension? TrySimplify(long scale, ReadOnlySpan<Dimension> dimensions)
    {
        if (dimensions.Length == 0 || dimensions.AsValueEnumerable().All(x => x.IsFixed))
        {
            for (var i = 0; i < dimensions.Length; i++)
            {
                scale *= dimensions[i].FixedValue;
            }

            return new DimConst(scale);
        }
        else if (dimensions.AsValueEnumerable().Any(x => x.IsUnknown))
        {
            return Unknown;
        }

        Dimension lhs = scale;
        for (var i = 0; i < dimensions.Length; i++)
        {
            var rhs = dimensions[i];
            lhs = (lhs, rhs) switch
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
                (DimProduct lhsProduct, DimProduct rhsProduct) => CreateWithSimplify(SpanUtility.Concat(lhsProduct.Operands, rhsProduct.Operands), lhsProduct.Scale * rhsProduct.Scale),
                (DimProduct lhsProduct, _) => CreateWithSimplify(SpanUtility.Concat(lhsProduct.Operands, [rhs]), lhsProduct.Scale),
                (_, DimProduct rhsProduct) => CreateWithSimplify(SpanUtility.Concat([lhs], rhsProduct.Operands), rhsProduct.Scale),
                (_, _) => CreateWithSimplify([lhs, rhs], 1),
            };
        }

        return lhs;
    }

    /// <inheritdoc/>
    protected override int GetHashCodeCore()
    {
        var hash = default(HashCode);
        hash.Add(Scale);
        foreach (var operand in Operands)
        {
            hash.Add(operand);
        }

        return hash.ToHashCode();
    }

    private static Dimension CreateWithSimplify(Dimension[] operands, long scale)
    {
        (var newScale, var pows) = DimHelpers.GetScaleAndPows(new DimProduct(operands, scale));
        return DimHelpers.Simplify(newScale, pows);
    }

    private ValueRange<double> InferRange()
    {
        var operands = Operands.ToArray().Append(Scale);
        var allMinMax = from lhs in operands
                        from rhs in operands
                        where !ReferenceEquals(lhs, rhs)
                        let lhsRange = lhs.Metadata.Range ?? ValueRange<double>.Full
                        let rhsRange = rhs.Metadata.Range ?? ValueRange<double>.Full
                        let ranges = new[] {
                            lhsRange.Min * rhsRange.Min,
                            lhsRange.Min * rhsRange.Max,
                            lhsRange.Max * rhsRange.Min,
                            lhsRange.Max * rhsRange.Max,
                     }
                        select new ValueRange<double>(ranges.Min(), ranges.Max());
        var min = allMinMax.MinBy(x => x.Min).Min;
        var max = allMinMax.MaxBy(x => x.Max).Max;
        return new ValueRange<double>(min, max);
    }
}

public sealed class DimSum : Dimension, IEquatable<DimSum?>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DimSum"/> class.
    /// </summary>
    /// <param name="operands">Operands.</param>
    /// <param name="bias">Bias.</param>
    public DimSum(Dimension[] operands, long bias = 0)
        : base(operands)
    {
        Bias = bias;
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets operands.
    /// </summary>
    public new ReadOnlySpan<Dimension> Operands => SpanUtility.UnsafeCast<BaseExpr, Dimension>(base.Operands);

    public long Bias { get; }

    public int Count => Operands.Length;

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimSum(this, context);

    public Dimension With(Dimension[]? operands = null, long? bias = null)
    {
        if (operands == null)
        {
            return new DimSum(Operands.ToArray(), bias ?? Bias);
        }

        return TrySimplify(bias ?? Bias, operands) ?? new DimSum(operands, bias ?? Bias);
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimSum);

    /// <inheritdoc/>
    public bool Equals(DimSum? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Operands.SequenceEqual(other.Operands);
    }

    public override string ToString()
    {
        var bias = Bias == 0 ? string.Empty : $"{Bias} + ";
        return $"({bias}{StringUtility.Join(" + ", Operands)})";
    }

    public override Dimension Simplify() => TrySimplify(Bias, Operands) ?? this;

    internal static Dimension? TrySimplify(long bias, ReadOnlySpan<Dimension> dimensions)
    {
        if (dimensions.Length == 0 || dimensions.AsValueEnumerable().All(x => x.IsFixed))
        {
            for (var i = 0; i < dimensions.Length; i++)
            {
                bias += dimensions[i].FixedValue;
            }

            return new DimConst(bias);
        }
        else if (dimensions.AsValueEnumerable().Any(x => x.IsUnknown))
        {
            return Unknown;
        }

        Dimension lhs = bias;
        for (var i = 0; i < dimensions.Length; i++)
        {
            var rhs = dimensions[i];
            lhs = (lhs, rhs) switch
            {
                (DimConst lhsConst, DimConst rhsConst) => lhsConst.Value + rhsConst.Value,
                (DimConst dimConst, _) when dimConst.Value == 0 => rhs,
                (_, DimConst dimConst) when dimConst.Value == 0 => lhs,
                (_, _) when lhs.IsUnknown || rhs.IsUnknown => Unknown,
                (DimSum lhsSum, DimSum rhsSum) => CreateWithSimplify(SpanUtility.Concat(lhsSum.Operands, rhsSum.Operands)),
                (DimSum lhsSum, _) => CreateWithSimplify(SpanUtility.Concat(lhsSum.Operands, [rhs])),
                (_, DimSum rhsSum) => CreateWithSimplify(SpanUtility.Concat([lhs], rhsSum.Operands)),
                (_, _) => CreateWithSimplify([lhs, rhs]),
            };
        }

        return lhs;
    }

    /// <inheritdoc/>
    protected override int GetHashCodeCore()
    {
        var hash = default(HashCode);
        foreach (var operand in Operands)
        {
            hash.Add(operand);
        }

        return hash.ToHashCode();
    }

    private static Dimension CreateWithSimplify(Dimension[] operands)
    {
        long bias = 0;
        var scales = new Dictionary<Dimension, long>(ReferenceEqualityComparer.Instance);

        foreach (var operand in operands)
        {
            if (operand is DimConst dimConst)
            {
                bias += dimConst.Value;
            }
            else
            {
                ref var value = ref CollectionsMarshal.GetValueRefOrAddDefault(scales, operand, out _);
                value += 1;
            }
        }

        var newOperands = scales.Select(kvp => kvp.Key * kvp.Value).ToArray();
        return (bias, newOperands.Length) switch
        {
            (_, 0) => new DimConst(bias),
            (0, 1) => newOperands[0],
            _ => new DimSum(newOperands, bias),
        };
    }

    private ValueRange<double> InferRange()
    {
        var min = (double)Bias;
        var max = (double)Bias;
        foreach (var operand in Operands)
        {
            var range = operand.Metadata.Range ?? ValueRange<double>.Full;
            min = System.Math.Min(min, range.Min);
            max = System.Math.Max(max, range.Max);
        }

        return new ValueRange<double>(min, max);
    }
}

public sealed class DimAbs : Dimension, IEquatable<DimAbs?>
{
    public DimAbs(Dimension operand)
        : base([operand])
    {
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets operand.
    /// </summary>
    public Dimension Operand => (Dimension)Operands[0];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimAbs(this, context);

    public DimAbs With(Dimension? operand = null) => new DimAbs(operand ?? Operand);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimAbs);

    /// <inheritdoc/>
    public bool Equals(DimAbs? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Operand.Equals(other.Operand);
    }

    public override string ToString() => $"|{Operand}|";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => Operand.GetHashCode();

    private ValueRange<double> InferRange()
    {
        if (Operand.Metadata.Range is ValueRange<double> operandRange)
        {
            var ranges = new[] {
                System.Math.Abs(operandRange.Min),
                System.Math.Abs(operandRange.Max),
            };
            return new ValueRange<double>(ranges.Min(), ranges.Max());
        }

        return ValueRange<double>.Full;
    }
}

public sealed class DimClamp : OpaqueDim, IEquatable<DimClamp?>
{
    public DimClamp(Dimension operand, Dimension minValue, Dimension maxValue)
        : base([operand, minValue, maxValue])
    {
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets operand.
    /// </summary>
    public Dimension Operand => (Dimension)Operands[0];

    /// <summary>
    /// Gets min.
    /// </summary>
    public Dimension MinValue => (Dimension)Operands[1];

    /// <summary>
    /// Gets max.
    /// </summary>
    public Dimension MaxValue => (Dimension)Operands[2];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimClamp(this, context);

    public DimClamp With(Dimension? operand = null, Dimension? minValue = null, Dimension? maxValue = null) =>
        new DimClamp(operand ?? Operand, minValue ?? MinValue, maxValue ?? MaxValue);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimClamp);

    /// <inheritdoc/>
    public bool Equals(DimClamp? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Operand.Equals(other.Operand) && MinValue.Equals(other.MinValue) && MaxValue.Equals(other.MaxValue);
    }

    public override Dimension Simplify()
    {
        var operandRange = Operand.Metadata.Range!.Value;
        var minValueRange = MinValue.Metadata.Range!.Value;
        var maxValueRange = MaxValue.Metadata.Range!.Value;
        if (operandRange.Max <= minValueRange.Min)
        {
            return MinValue;
        }
        else if (operandRange.Min >= maxValueRange.Max)
        {
            return MaxValue;
        }
        else if (operandRange.Min >= minValueRange.Min && operandRange.Max <= maxValueRange.Max)
        {
            return Operand;
        }

        return this;
    }

    public override string ToString() => $"clamp({Operand}, {MinValue}, {MaxValue})";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Operand, MinValue, MaxValue);

    private ValueRange<double> InferRange()
    {
        var operandRange = Operand.Metadata.Range!.Value;
        var min = System.Math.Max(operandRange.Min, MinValue.Metadata.Range!.Value.Min);
        var max = System.Math.Min(operandRange.Max, MaxValue.Metadata.Range!.Value.Max);
        return new ValueRange<double>(min, max);
    }
}

public sealed class DimCompareAndSelect : OpaqueDim, IEquatable<DimCompareAndSelect?>
{
    public DimCompareAndSelect(Dimension value, Dimension expected, Dimension trueValue, Dimension falseValue)
        : base([value, expected, trueValue, falseValue])
    {
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets value.
    /// </summary>
    public Dimension Value => (Dimension)Operands[0];

    /// <summary>
    /// Gets expected.
    /// </summary>
    public Dimension Expected => (Dimension)Operands[1];

    /// <summary>
    /// Gets true value.
    /// </summary>
    public Dimension TrueValue => (Dimension)Operands[2];

    /// <summary>
    /// Gets false value.
    /// </summary>
    public Dimension FalseValue => (Dimension)Operands[3];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimCompareAndSelect(this, context);

    public DimCompareAndSelect With(Dimension? value = null, Dimension? expected = null, Dimension? trueValue = null, Dimension? falseValue = null) =>
        new DimCompareAndSelect(value ?? Value, expected ?? Expected, trueValue ?? TrueValue, falseValue ?? FalseValue);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimCompareAndSelect);

    /// <inheritdoc/>
    public bool Equals(DimCompareAndSelect? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Value.Equals(other.Value) && Expected.Equals(other.Expected) && TrueValue.Equals(other.TrueValue) && FalseValue.Equals(other.FalseValue);
    }

    public override Dimension Simplify()
    {
        if (Value == FalseValue && Expected == TrueValue)
        {
            if (Value.IsFixed)
            {
                return new DimConst(Value.FixedValue);
            }

            return Value;
        }

        return this;
    }

    public override string ToString() => $"({Value} == {Expected} ? {TrueValue} : {FalseValue})";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Value, Expected, TrueValue, FalseValue);

    private ValueRange<double> InferRange()
    {
        var trueValueRange = TrueValue.Metadata.Range ?? ValueRange<double>.Full;
        var falseValueRange = FalseValue.Metadata.Range ?? ValueRange<double>.Full;
        var min = System.Math.Min(trueValueRange.Min, falseValueRange.Min);
        var max = System.Math.Max(trueValueRange.Max, falseValueRange.Max);
        return new ValueRange<double>(min, max);
    }
}

public sealed class DimMin : OpaqueDim, IEquatable<DimMin?>
{
    public DimMin(params Dimension[] operands)
        : base(operands)
    {
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets operands.
    /// </summary>
    public new ReadOnlySpan<Dimension> Operands => SpanUtility.UnsafeCast<BaseExpr, Dimension>(base.Operands);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimMin(this, context);

    public Dimension With(Dimension[]? operands = null)
    {
        if (operands == null)
        {
            return new DimMin(Operands.ToArray());
        }

        return Dimension.Min(operands);
    }

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimMin);

    /// <inheritdoc/>
    public bool Equals(DimMin? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Operands.SequenceEqual(other.Operands);
    }

    public override Dimension Simplify() => TrySimplify(Operands) ?? this;

    public override string ToString() => $"min({StringUtility.Join(", ", Operands)})";

    internal static Dimension? TrySimplify(ReadOnlySpan<Dimension> dimensions)
    {
        if (dimensions.Length == 0)
        {
            throw new ArgumentException("At least one dimension is required.");
        }

        if (dimensions.AsValueEnumerable().All(x => x.IsFixed))
        {
            var min = dimensions[0].FixedValue;
            for (var i = 1; i < dimensions.Length; i++)
            {
                min = System.Math.Min(min, dimensions[i].FixedValue);
            }

            return new DimConst(min);
        }
        else if (dimensions.AsValueEnumerable().Any(x => x.IsUnknown))
        {
            return Unknown;
        }

        return null;
    }

    /// <inheritdoc/>
    protected override int GetHashCodeCore()
    {
        var hash = default(HashCode);
        foreach (var operand in Operands)
        {
            hash.Add(operand);
        }

        return hash.ToHashCode();
    }

    private ValueRange<double> InferRange()
    {
        var min = double.MaxValue;
        var max = double.MinValue;
        foreach (var operand in Operands)
        {
            var range = operand.Metadata.Range!.Value;
            min = System.Math.Min(min, range.Min);
            max = System.Math.Min(max, range.Max);
        }

        return new ValueRange<double>(min, max);
    }
}

public sealed class DimMax : OpaqueDim, IEquatable<DimMax?>
{
    public DimMax(params Dimension[] operands)
        : base(operands)
    {
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets operands.
    /// </summary>
    public new ReadOnlySpan<Dimension> Operands => SpanUtility.UnsafeCast<BaseExpr, Dimension>(base.Operands);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimMax(this, context);

    public DimMax With(Dimension[]? operands = null) => new DimMax(operands ?? Operands.ToArray());

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimMax);

    /// <inheritdoc/>
    public bool Equals(DimMax? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Operands.SequenceEqual(other.Operands);
    }

    public override string ToString() => $"max({StringUtility.Join(", ", Operands)})";

    /// <inheritdoc/>
    protected override int GetHashCodeCore()
    {
        var hash = default(HashCode);
        foreach (var operand in Operands)
        {
            hash.Add(operand);
        }

        return hash.ToHashCode();
    }

    private ValueRange<double> InferRange()
    {
        var min = double.MaxValue;
        var max = double.MinValue;
        foreach (var operand in Operands)
        {
            var range = operand.Metadata.Range ?? ValueRange<double>.Full;
            min = System.Math.Max(min, range.Min);
            max = System.Math.Max(max, range.Max);
        }

        return new ValueRange<double>(min, max);
    }
}

public sealed class DimPositive : OpaqueDim, IEquatable<DimPositive?>
{
    public DimPositive(Dimension operand, Dimension extent)
        : base([operand, extent])
    {
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets operand.
    /// </summary>
    public Dimension Operand => (Dimension)Operands[0];

    /// <summary>
    /// Gets extent.
    /// </summary>
    public Dimension Extent => (Dimension)Operands[1];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimPositive(this, context);

    public DimPositive With(Dimension? operand = null, Dimension? extent = null) => new DimPositive(operand ?? Operand, extent ?? Extent);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimPositive);

    /// <inheritdoc/>
    public bool Equals(DimPositive? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Operand.Equals(other.Operand) && Extent.Equals(other.Extent);
    }

    public override string ToString() => $"positive({Operand}, {Extent})";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Operand, Extent);

    private ValueRange<double> InferRange()
    {
        if (Operand.Metadata.Range is ValueRange<double> operandRange && Extent.Metadata.Range is ValueRange<double> extentRange)
        {
            var min = operandRange.Min < 0 ? 0 : operandRange.Min;
            var max = operandRange.Max >= 0 ? System.Math.Min(operandRange.Max, extentRange.Max) : extentRange.Max;
            return new ValueRange<double>(min, max);
        }

        return ValueRange<double>.Full;
    }
}

public sealed class DimAt : OpaqueDim, IEquatable<DimAt?>
{
    public DimAt(Shape shape, Dimension index)
        : base([shape, index])
    {
        Metadata.Range = InferRange();
    }

    public override DimensionKind Kind => DimensionKind.Dynamic;

    /// <summary>
    /// Gets shape.
    /// </summary>
    public Shape Shape => (Shape)Operands[0];

    /// <summary>
    /// Gets index.
    /// </summary>
    public Dimension Index => (Dimension)Operands[1];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimAt(this, context);

    public DimAt With(Shape? shape = null, Dimension? index = null) => new DimAt(shape ?? Shape, index ?? Index);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimAt);

    /// <inheritdoc/>
    public bool Equals(DimAt? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Shape.Equals(other.Shape) && Index.Equals(other.Index);
    }

    public override string ToString() => $"at({Shape}, {Index})";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Shape, Index);

    private ValueRange<double> InferRange()
    {
        if (Shape.Metadata.Range is ValueRange<double> shapeRange)
        {
            return shapeRange;
        }

        return ValueRange<double>.Full;
    }
}

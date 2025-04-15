// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections;
using System.Collections.Generic;
using Nncase.Utilities;

namespace Nncase.IR;

/// <summary>
/// Base class for dimension expression.
/// </summary>
public abstract class DimExpr : Expr
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DimExpr"/> class.
    /// </summary>
    /// <param name="operands">Operands.</param>
    public DimExpr(Expr[] operands)
        : base(operands)
    {
    }

    public static implicit operator DimExpr(long value) => new DimConst(value);

    public static implicit operator DimExpr(string name) => new DimVar(name);
}

public sealed class UnknownDim : DimExpr, IEquatable<UnknownDim?>
{
    public UnknownDim()
        : base(Array.Empty<Expr>())
    {
    }

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitUnknownDim(this, context);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as UnknownDim);

    /// <inheritdoc/>
    public bool Equals(UnknownDim? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null;
    }

    public override string ToString() => "?";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => 0;
}

public sealed class DimVar : DimExpr, IEquatable<DimVar?>
{
    private static int _globalVarIndex;

    /// <summary>
    /// Initializes a new instance of the <see cref="DimVar"/> class.
    /// </summary>
    /// <param name="name">Name.</param>
    public DimVar(string name)
        : base(Array.Empty<Expr>())
    {
        GlobalVarIndex = GetNextId();
        Name = name;
    }

    /// <summary>
    /// Gets the global var index.
    /// </summary>
    public int GlobalVarIndex { get; }

    /// <summary>
    /// Gets name.
    /// </summary>
    public string Name { get; }

    /// <summary>
    /// Create a dim var.
    /// </summary>
    public static implicit operator DimVar(string name) => new DimVar(name);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimVar(this, context);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as Var);

    /// <inheritdoc/>
    public bool Equals(DimVar? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && GlobalVarIndex == other.GlobalVarIndex;
    }

    public override string ToString() => $"{Name}#{GlobalVarIndex}";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(GlobalVarIndex);

    private static int GetNextId()
    {
        return Interlocked.Increment(ref _globalVarIndex);
    }
}

public sealed class DimConst : DimExpr, IEquatable<DimConst?>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DimConst"/> class.
    /// </summary>
    /// <param name="value">Value.</param>
    public DimConst(long value)
        : base(Array.Empty<Expr>())
    {
        Value = value;
    }

    /// <summary>
    /// Gets value.
    /// </summary>
    public long Value { get; }

    public static implicit operator DimConst(long value) => new DimConst(value);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimConst(this, context);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimConst);

    /// <inheritdoc/>
    public bool Equals(DimConst? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Value == other.Value;
    }

    public override string ToString() => Value.ToString();

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Value);
}

public sealed class ScaledDim : DimExpr, IEquatable<ScaledDim?>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="ScaledDim"/> class.
    /// </summary>
    /// <param name="dim">Dim.</param>
    /// <param name="scale">Scale.</param>
    public ScaledDim(DimExpr dim, long scale)
        : base([dim])
    {
        Scale = scale;
    }

    /// <summary>
    /// Gets dim.
    /// </summary>
    public DimExpr Dim => (DimExpr)Operands[0];

    /// <summary>
    /// Gets scale.
    /// </summary>
    public long Scale { get; }

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitScaledDim(this, context);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as ScaledDim);

    /// <inheritdoc/>
    public bool Equals(ScaledDim? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Dim.Equals(other.Dim) && Scale == other.Scale;
    }

    /// <inheritdoc/>
    public override string ToString() =>
        Scale switch
        {
            1 => Dim.ToString(),
            -1 => $"-{Dim}",
            _ => $"({Scale} * {Dim})",
        };

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Dim, Scale);
}

public sealed class DimFraction : DimExpr, IEquatable<DimFraction?>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DimFraction"/> class.
    /// </summary>
    /// <param name="numerator">Numerator.</param>
    /// <param name="denominator">Denominator.</param>
    public DimFraction(DimExpr numerator, DimExpr denominator)
        : base([numerator, denominator])
    {
    }

    /// <summary>
    /// Gets numerator.
    /// </summary>
    public DimExpr Numerator => (DimExpr)Operands[0];

    /// <summary>
    /// Gets denominator.
    /// </summary>
    public DimExpr Denominator => (DimExpr)Operands[1];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimFraction(this, context);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimFraction);

    /// <inheritdoc/>
    public bool Equals(DimFraction? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Numerator.Equals(other.Numerator) && Denominator.Equals(other.Denominator);
    }

    public override string ToString() => $"({Numerator} / {Denominator})";

    /// <inheritdoc/>
    protected override int GetHashCodeCore() => HashCode.Combine(Numerator, Denominator);
}

public sealed class DimProduct : DimExpr, IEquatable<DimProduct?>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DimProduct"/> class.
    /// </summary>
    /// <param name="operands">Operands.</param>
    public DimProduct(params DimExpr[] operands)
        : base(operands)
    {
    }

    /// <summary>
    /// Gets operands.
    /// </summary>
    public new ReadOnlySpan<DimExpr> Operands => SpanUtility.UnsafeCast<Expr, DimExpr>(base.Operands);

    public int Count => Operands.Length;

    public new DimExpr this[int index] => Operands[index];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimProduct(this, context);

    /// <inheritdoc/>
    public override bool Equals(object? obj) => Equals(obj as DimProduct);

    /// <inheritdoc/>
    public bool Equals(DimProduct? other)
    {
        if (ReferenceEquals(this, other))
        {
            return true;
        }

        return other is not null && Operands.SequenceEqual(other.Operands);
    }

    public override string ToString() => $"({StringUtility.Join(" * ", Operands)})";

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
}

public sealed class DimSum : DimExpr, IEquatable<DimSum?>
{
    /// <summary>
    /// Initializes a new instance of the <see cref="DimSum"/> class.
    /// </summary>
    /// <param name="operands">Operands.</param>
    public DimSum(params DimExpr[] operands)
        : base(operands)
    {
    }

    /// <summary>
    /// Gets operands.
    /// </summary>
    public new ReadOnlySpan<DimExpr> Operands => SpanUtility.UnsafeCast<Expr, DimExpr>(base.Operands);

    public int Count => Operands.Length;

    public new DimExpr this[int index] => Operands[index];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) =>
        functor.VisitDimSum(this, context);

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

    public override string ToString() => $"({StringUtility.Join(" + ", Operands)})";

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
}

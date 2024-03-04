// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Xml.Linq;
using DryIoc.ImTools;
using Microsoft.Toolkit.HighPerformance;
using Nncase.IR;

namespace Nncase.IR.Affine;

public enum AffineDivBinaryOp
{
    FloorDiv,
    CeilDiv,
    Mod,
}

public abstract class AffineExpr : Expr
{
    internal AffineExpr(Expr[] operands)
        : base(operands)
    {
    }

    public static implicit operator AffineExpr(long value) => new AffineConstant(value);

    public static AffineExpr operator -(AffineExpr value) => -1 * value;

    public static AffineExpr operator +(AffineExpr lhs, AffineExpr rhs) => new AffineAddBinary(lhs, rhs);

    public static AffineExpr operator -(AffineExpr lhs, AffineExpr rhs) => lhs + -rhs;

    public static AffineExpr operator *(AffineSymbolBase lhs, AffineExpr rhs) => new AffineMulBinary(lhs, rhs);

    public static AffineExpr operator *(AffineConstant lhs, AffineExpr rhs) => new AffineMulBinary(lhs, rhs);

    public static AffineExpr operator %(AffineExpr lhs, AffineSymbolBase rhs) => new AffineDivBinary(AffineDivBinaryOp.Mod, lhs, rhs);

    public static AffineExpr operator %(AffineExpr lhs, AffineConstant rhs) => new AffineDivBinary(AffineDivBinaryOp.Mod, lhs, rhs);

    /// <summary>
    /// Accept a <see cref="AffineExprVisitor{TExprResult, TContext}"/>.
    /// </summary>
    /// <typeparam name="TExprResult">Result type of visiting expressions.</typeparam>
    /// <typeparam name="TContext">Visit context.</typeparam>
    /// <param name="functor">Expression functor.</param>
    /// <param name="context">Context.</param>
    /// <returns>Visit result.</returns>
    public abstract TExprResult Accept<TExprResult, TContext>(AffineExprVisitor<TExprResult, TContext> functor, TContext context);

    internal AffineExpr ReplaceDomains(ReadOnlySpan<AffineRange> newDomains)
    {
        return this switch
        {
            AffineDim e when e.Position < newDomains.Length => newDomains[e.Position].Offset,
            AffineExtent e when e.Position < newDomains.Length => newDomains[e.Position].Extent,
            AffineAddBinary e => new AffineAddBinary(e.Lhs.ReplaceDomains(newDomains), e.Rhs.ReplaceDomains(newDomains)),
            AffineMulBinary e => new AffineMulBinary((AffineSymbolBase)e.Lhs.ReplaceDomains(newDomains), e.Rhs.ReplaceDomains(newDomains)),
            AffineDivBinary e => new AffineDivBinary(e.BinaryOp, e.Lhs.ReplaceDomains(newDomains), (AffineSymbolBase)e.Rhs.ReplaceDomains(newDomains)),
            _ => this,
        };
    }

    internal Expr Apply(ReadOnlySpan<Expr> dims, ReadOnlySpan<Expr> extents, IReadOnlyDictionary<AffineSymbol, Expr> symbols)
    {
        static Expr ApplyDivBinary(AffineDivBinaryOp binaryOp, Expr lhs, Expr rhs) =>
            binaryOp switch
            {
                AffineDivBinaryOp.FloorDiv => F.Math.FloorDiv(lhs, rhs),
                AffineDivBinaryOp.CeilDiv => F.Math.CeilDiv(lhs, rhs),
                AffineDivBinaryOp.Mod => F.Math.Mod(lhs, rhs),
                _ => throw new UnreachableException(),
            };

        return this switch
        {
            AffineConstant e => e.Value,
            AffineExtent e => extents[e.Position],
            AffineDim e => dims[e.Position],
            AffineSymbol e => symbols[e],
            AffineAddBinary e => e.Lhs.Apply(dims, extents, symbols) + e.Rhs.Apply(dims, extents, symbols),
            AffineMulBinary e => e.Lhs.Apply(dims, extents, symbols) * e.Rhs.Apply(dims, extents, symbols),
            AffineDivBinary e => ApplyDivBinary(e.BinaryOp, e.Lhs.Apply(dims, extents, symbols), e.Rhs.Apply(dims, extents, symbols)),
            _ => throw new UnreachableException(),
        };
    }

    internal string GetDisplayString(ReadOnlySpan<AffineSymbol> symbols)
    {
        return this switch
        {
            AffineConstant e => e.Value.ToString(),
            AffineExtent e => $"t{e.Position}",
            AffineDim e => $"d{e.Position}",
            AffineSymbol e => $"d{symbols.IndexOf(e)}",
            AffineAddBinary e => $"({e.Lhs.GetDisplayString(symbols)} + {e.Rhs.GetDisplayString(symbols)})",
            AffineMulBinary e => $"({e.Lhs.GetDisplayString(symbols)} * {e.Rhs.GetDisplayString(symbols)})",
            AffineDivBinary e => $"({e.Lhs.GetDisplayString(symbols)} {F.Affine.ToString(e.BinaryOp)} {e.Rhs.GetDisplayString(symbols)})",
            _ => throw new UnreachableException(),
        };
    }
}

public sealed class AffineDim : AffineExpr
{
    public AffineDim(int position)
        : base(Array.Empty<Expr>())
    {
        Position = position;
    }

    public int Position { get; }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TContext>(AffineExprVisitor<TExprResult, TContext> functor, TContext context) => functor.VisitAffineDim(this, context);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitAffineDim(this, context);

    public AffineDim With(int? position = null) => new AffineDim(position ?? Position);

    public override string ToString() => $"d{Position}";

    protected override int GetHashCodeCore() => HashCode.Combine(Position);
}

public abstract class AffineSymbolBase : AffineExpr
{
    public AffineSymbolBase()
        : base(Array.Empty<Expr>())
    {
    }
}

public sealed class AffineExtent : AffineSymbolBase
{
    public AffineExtent(int position)
    {
        Position = position;
    }

    public int Position { get; }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TContext>(AffineExprVisitor<TExprResult, TContext> functor, TContext context) => functor.VisitAffineExtent(this, context);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitAffineExtent(this, context);

    public AffineExtent With(int? position = null) => new AffineExtent(position ?? Position);

    public override string ToString() => $"t{Position}";

    protected override int GetHashCodeCore() => HashCode.Combine(Position);
}

public sealed class AffineSymbol : AffineSymbolBase
{
    public AffineSymbol(string name)
    {
        Name = name;
    }

    public string Name { get; }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TContext>(AffineExprVisitor<TExprResult, TContext> functor, TContext context) => functor.VisitAffineSymbol(this, context);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitAffineSymbol(this, context);

    public AffineSymbol With(string? name = null) => new AffineSymbol(name ?? Name);

    public override string ToString() => Name;
}

public sealed class AffineConstant : AffineSymbolBase
{
    public AffineConstant(long value)
    {
        Value = value;
    }

    public long Value { get; }

    public static implicit operator AffineConstant(long value) => new AffineConstant(value);

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TContext>(AffineExprVisitor<TExprResult, TContext> functor, TContext context) => functor.VisitAffineConstant(this, context);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitAffineConstant(this, context);

    public AffineConstant With(long? value = null) => new AffineConstant(value ?? Value);

    public override string ToString() => Value.ToString();
}

public sealed class AffineAddBinary : AffineExpr
{
    public AffineAddBinary(AffineExpr lhs, AffineExpr rhs)
        : base(new Expr[] { lhs, rhs })
    {
    }

    public AffineExpr Lhs => (AffineExpr)Operands[0];

    public AffineExpr Rhs => (AffineExpr)Operands[1];

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TContext>(AffineExprVisitor<TExprResult, TContext> functor, TContext context) => functor.VisitAffineAddBinary(this, context);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitAffineAddBinary(this, context);

    public AffineAddBinary With(AffineExpr? lhs = null, AffineExpr? rhs = null) => new AffineAddBinary(lhs ?? Lhs, rhs ?? Rhs);

    public override string ToString() => $"({Lhs} + {Rhs})";
}

public sealed class AffineMulBinary : AffineExpr
{
    public AffineMulBinary(AffineSymbolBase lhs, AffineExpr rhs)
        : base(new Expr[] { lhs, rhs })
    {
    }

    public AffineSymbolBase Lhs => (AffineSymbolBase)Operands[0];

    public AffineExpr Rhs => (AffineExpr)Operands[1];

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TContext>(AffineExprVisitor<TExprResult, TContext> functor, TContext context) => functor.VisitAffineMulBinary(this, context);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitAffineMulBinary(this, context);

    public AffineMulBinary With(AffineSymbolBase? lhs = null, AffineExpr? rhs = null) => new AffineMulBinary(lhs ?? Lhs, rhs ?? Rhs);

    public override string ToString() => $"({Lhs} * {Rhs})";
}

public sealed class AffineDivBinary : AffineExpr
{
    public AffineDivBinary(AffineDivBinaryOp binaryOp, AffineExpr lhs, AffineSymbolBase rhs)
        : base(new Expr[] { lhs, rhs })
    {
        BinaryOp = binaryOp;
    }

    public AffineDivBinaryOp BinaryOp { get; }

    public AffineExpr Lhs => (AffineExpr)Operands[0];

    public AffineSymbolBase Rhs => (AffineSymbolBase)Operands[1];

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TContext>(AffineExprVisitor<TExprResult, TContext> functor, TContext context) => functor.VisitAffineDivBinary(this, context);

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitAffineDivBinary(this, context);

    public AffineDivBinary With(AffineDivBinaryOp? binaryOp = null, AffineExpr? lhs = null, AffineSymbolBase? rhs = null) => new AffineDivBinary(binaryOp ?? BinaryOp, lhs ?? Lhs, rhs ?? Rhs);

    public override string ToString() => $"({Lhs} {F.Affine.ToString(BinaryOp)} {Rhs})";
}

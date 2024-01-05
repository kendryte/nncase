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
using Nncase.IR;

namespace Nncase.Tiling;

public enum AffineDivBinaryOp
{
    FloorDiv,
    CeilDiv,
    Mod,
}

public abstract class AffineExpr
{
    public static implicit operator AffineExpr(long value) => new AffineConstantExpr(value);

    public static AffineExpr operator -(AffineExpr value) => -1 * value;

    public static AffineExpr operator +(AffineExpr lhs, AffineExpr rhs) => new AffineAddBinaryExpr(lhs, rhs);

    public static AffineExpr operator -(AffineExpr lhs, AffineExpr rhs) => lhs + (-rhs);

    public static AffineExpr operator *(AffineConstantExpr lhs, AffineExpr rhs) => new AffineMulBinaryExpr(lhs, rhs);

    public static AffineExpr operator *(AffineSymbolExpr lhs, AffineExpr rhs) => new AffineMulBinaryExpr(lhs, rhs);

    public static AffineExpr operator %(AffineExpr lhs, AffineConstantExpr rhs) => new AffineDivBinaryExpr(AffineDivBinaryOp.Mod, lhs, rhs);

    public static AffineExpr operator %(AffineExpr lhs, AffineSymbolExpr rhs) => new AffineDivBinaryExpr(AffineDivBinaryOp.Mod, lhs, rhs);

    /// <summary>
    /// Accept a <see cref="AffineExprVisitor{TExprResult, TContext}"/>.
    /// </summary>
    /// <typeparam name="TExprResult">Result type of visiting expressions.</typeparam>
    /// <typeparam name="TContext">Visit context.</typeparam>
    /// <param name="functor">Expression functor.</param>
    /// <param name="context">Context.</param>
    /// <returns>Visit result.</returns>
    public abstract TExprResult Accept<TExprResult, TContext>(AffineExprVisitor<TExprResult, TContext> functor, TContext context);

    internal AffineExpr ReplaceDims(IReadOnlyList<AffineExpr> newDims)
    {
        return this switch
        {
            AffineDimExpr e when e.Position < newDims.Count => newDims[e.Position],
            AffineAddBinaryExpr e => new AffineAddBinaryExpr(e.Lhs.ReplaceDims(newDims), e.Rhs.ReplaceDims(newDims)),
            AffineMulBinaryExpr e => new AffineMulBinaryExpr(ReplaceDims(e.Lhs, newDims), e.Rhs.ReplaceDims(newDims)),
            AffineDivBinaryExpr e => new AffineDivBinaryExpr(e.BinaryOp, e.Lhs.ReplaceDims(newDims), ReplaceDims(e.Rhs, newDims)),
            _ => this,
        };
    }

    internal AffineExpr Compose(AffineMap affineMap)
        => ReplaceDims(affineMap.Results);

    internal Expr Apply(IReadOnlyList<Expr> dims, IReadOnlyDictionary<AffineSymbolExpr, Expr> symbols)
    {
        static Expr ApplyDivBinary(AffineDivBinaryOp binaryOp, Expr lhs, Expr rhs) =>
            binaryOp switch
            {
                AffineDivBinaryOp.FloorDiv => IR.F.Math.FloorDiv(lhs, rhs),
                AffineDivBinaryOp.CeilDiv => IR.F.Math.CeilDiv(lhs, rhs),
                AffineDivBinaryOp.Mod => IR.F.Math.Mod(lhs, rhs),
                _ => throw new UnreachableException(),
            };

        return this switch
        {
            AffineConstantExpr e => e.Value,
            AffineDimExpr e => dims[e.Position],
            AffineSymbolExpr e => symbols[e],
            AffineAddBinaryExpr e => e.Lhs.Apply(dims, symbols) + e.Rhs.Apply(dims, symbols),
            AffineMulBinaryExpr e => Apply(e.Lhs, dims, symbols) * e.Rhs.Apply(dims, symbols),
            AffineDivBinaryExpr e => ApplyDivBinary(e.BinaryOp, e.Lhs.Apply(dims, symbols), Apply(e.Rhs, dims, symbols)),
            _ => throw new UnreachableException(),
        };
    }

    internal string GetDisplayString(AffineSymbolExpr[] symbols)
    {
        return this switch
        {
            AffineConstantExpr e => e.Value.ToString(),
            AffineDimExpr e => $"d{e.Position}",
            AffineSymbolExpr e => $"d{Array.IndexOf(symbols, e)}",
            AffineAddBinaryExpr e => $"({e.Lhs.GetDisplayString(symbols)} + {e.Rhs.GetDisplayString(symbols)})",
            AffineMulBinaryExpr e => $"({GetDisplayString(e.Lhs, symbols)} * {e.Rhs.GetDisplayString(symbols)})",
            AffineDivBinaryExpr e => $"({e.Lhs.GetDisplayString(symbols)} {Affine.ToString(e.BinaryOp)} {GetDisplayString(e.Rhs, symbols)})",
            _ => throw new UnreachableException(),
        };
    }

    private static Either<AffineConstantExpr, AffineSymbolExpr> ReplaceDims(Either<AffineConstantExpr, AffineSymbolExpr> expr, IReadOnlyList<AffineExpr> newDims)
    {
        return expr.Match(x => (Either<AffineConstantExpr, AffineSymbolExpr>)x, x => (AffineSymbolExpr)x.ReplaceDims(newDims));
    }

    private static Expr Apply(Either<AffineConstantExpr, AffineSymbolExpr> expr, IReadOnlyList<Expr> dims, IReadOnlyDictionary<AffineSymbolExpr, Expr> symbols)
    {
        return expr.Match(x => x.Value, x => symbols[x]);
    }

    private static string GetDisplayString(Either<AffineConstantExpr, AffineSymbolExpr> expr, AffineSymbolExpr[] symbols)
    {
        return expr.Match(x => x.Value.ToString(), x => $"d{Array.IndexOf(symbols, x)}");
    }
}

public sealed class AffineDimExpr : AffineExpr
{
    public AffineDimExpr(int position)
    {
        Position = position;
    }

    public int Position { get; }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TContext>(AffineExprVisitor<TExprResult, TContext> functor, TContext context) => functor.VisitAffineDimExpr(this, context);

    public override string ToString() => $"d{Position}";
}

public sealed class AffineSymbolExpr : AffineExpr
{
    public AffineSymbolExpr(string name)
    {
        Name = name;
    }

    public string Name { get; }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TContext>(AffineExprVisitor<TExprResult, TContext> functor, TContext context) => functor.VisitAffineSymbolExpr(this, context);

    public override string ToString() => Name;
}

public sealed class AffineConstantExpr : AffineExpr
{
    public AffineConstantExpr(long value)
    {
        Value = value;
    }

    public long Value { get; }

    public static implicit operator AffineConstantExpr(long value) => new AffineConstantExpr(value);

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TContext>(AffineExprVisitor<TExprResult, TContext> functor, TContext context) => functor.VisitAffineConstantExpr(this, context);

    public override string ToString() => Value.ToString();
}

public sealed class AffineAddBinaryExpr : AffineExpr
{
    public AffineAddBinaryExpr(AffineExpr lhs, AffineExpr rhs)
    {
        Lhs = lhs;
        Rhs = rhs;
    }

    public AffineExpr Lhs { get; }

    public AffineExpr Rhs { get; }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TContext>(AffineExprVisitor<TExprResult, TContext> functor, TContext context) => functor.VisitAffineAddBinaryExpr(this, context);

    public override string ToString() => $"({Lhs} + {Rhs})";
}

public sealed class AffineMulBinaryExpr : AffineExpr
{
    public AffineMulBinaryExpr(Either<AffineConstantExpr, AffineSymbolExpr> lhs, AffineExpr rhs)
    {
        Lhs = lhs;
        Rhs = rhs;
    }

    public Either<AffineConstantExpr, AffineSymbolExpr> Lhs { get; }

    public AffineExpr Rhs { get; }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TContext>(AffineExprVisitor<TExprResult, TContext> functor, TContext context) => functor.VisitAffineMulBinaryExpr(this, context);

    public override string ToString() => $"({Lhs} * {Rhs})";
}

public sealed class AffineDivBinaryExpr : AffineExpr
{
    public AffineDivBinaryExpr(AffineDivBinaryOp binaryOp, AffineExpr lhs, Either<AffineConstantExpr, AffineSymbolExpr> rhs)
    {
        BinaryOp = binaryOp;
        Lhs = lhs;
        Rhs = rhs;
    }

    public AffineDivBinaryOp BinaryOp { get; }

    public AffineExpr Lhs { get; }

    public Either<AffineConstantExpr, AffineSymbolExpr> Rhs { get; }

    /// <inheritdoc/>
    public override TExprResult Accept<TExprResult, TContext>(AffineExprVisitor<TExprResult, TContext> functor, TContext context) => functor.VisitAffineDivBinaryExpr(this, context);

    public override string ToString() => $"({Lhs} {Affine.ToString(BinaryOp)} {Rhs})";
}

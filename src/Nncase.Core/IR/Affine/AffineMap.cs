// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using NetFabric.Hyperlinq;
using Nncase.Utilities;

namespace Nncase.IR.Affine;

public sealed class AffineDomain : Expr
{
    public AffineDomain(AffineDim offset, AffineExtent extent)
        : base(new Expr[] { offset, extent })
    {
    }

    public AffineDim Offset => (AffineDim)Operands[0];

    public AffineExtent Extent => (AffineExtent)Operands[1];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitAffineDomain(this, context);

    public AffineDomain With(AffineDim? offset = null, AffineExtent? extent = null)
        => new AffineDomain(offset ?? Offset, extent ?? Extent);

    public override string ToString() => $"({Offset}, {Extent})";
}

public sealed class AffineRange : Expr
{
    public AffineRange(AffineExpr offset, AffineExpr extent)
        : base(new Expr[] { offset, extent })
    {
    }

    public AffineExpr Offset => (AffineExpr)Operands[0];

    public AffineExpr Extent => (AffineExpr)Operands[1];

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitAffineRange(this, context);

    public AffineRange With(AffineExpr? offset = null, AffineExpr? extent = null)
        => new AffineRange(offset ?? Offset, extent ?? Extent);

    public (Expr Offset, Expr Extent) Apply(ReadOnlySpan<Expr> dims, ReadOnlySpan<Expr> extents, IReadOnlyDictionary<AffineSymbol, Expr>? symbols = null)
    {
        var offset = Offset.Apply(dims, extents, symbols);
        var extent = Extent.Apply(dims, extents, symbols);
        return (offset, extent);
    }

    public ValueRange<long> Apply(ReadOnlySpan<long> dims, ReadOnlySpan<long> extents, IReadOnlyDictionary<AffineSymbol, long>? symbols = null)
    {
        var offset = Offset.Apply(dims, extents, symbols);
        var extent = Extent.Apply(dims, extents, symbols);
        return (offset, extent);
    }

    internal string GetDisplayString(ReadOnlySpan<AffineSymbol> symbols)
        => $"({Offset.GetDisplayString(symbols)}, {Extent.GetDisplayString(symbols)})";

    internal AffineRange ReplaceDomains(ReadOnlySpan<AffineRange> newDomains)
        => new AffineRange(Offset.ReplaceDomainsAndSymbols(newDomains, Array.Empty<AffineSymbol>()), Extent.ReplaceDomainsAndSymbols(newDomains, Array.Empty<AffineSymbol>()));
}

public sealed class AffineMap : Expr
{
    private readonly int _domainsCount;
    private readonly int _symbolsCount;

    public AffineMap(ReadOnlySpan<AffineDomain> domains, ReadOnlySpan<AffineSymbol> symbols, ReadOnlySpan<AffineRange> results)
        : base(domains.ToArray().AsEnumerable<Expr>().Concat(symbols.ToArray()).Concat(results.ToArray()))
    {
        _domainsCount = domains.Length;
        _symbolsCount = symbols.Length;
    }

    public ReadOnlySpan<AffineDomain> Domains => SpanUtility.UnsafeCast<Expr, AffineDomain>(Operands.Slice(0, _domainsCount));

    public ReadOnlySpan<AffineSymbol> Symbols => SpanUtility.UnsafeCast<Expr, AffineSymbol>(Operands.Slice(_domainsCount, _symbolsCount));

    public ReadOnlySpan<AffineRange> Results => SpanUtility.UnsafeCast<Expr, AffineRange>(Operands.Slice(_domainsCount + _symbolsCount));

    public static AffineMap operator *(AffineMap lhs, AffineMap rhs)
    {
        if (lhs.Results.Length != rhs.Domains.Length)
        {
            throw new ArgumentException("Cannot compose AffineMaps with mismatching dimensions and results.");
        }

        var results = rhs.Results.AsValueEnumerable().Select(x => x.ReplaceDomains(lhs.Results)).ToArray();
        var symbols = lhs.Symbols.ToArray().Concat(rhs.Symbols.ToArray()).ToArray();
        return new AffineMap(lhs.Domains, symbols, results);
    }

    public static AffineMap FromCallable(Func<AffineDomain[], AffineSymbol[], AffineRange[]> func, int dimsCount, int symbolsCount = 0)
    {
        var domains = F.Affine.Domains(dimsCount);
        var symbols = F.Affine.Symbols(symbolsCount);
        var results = func(domains, symbols);
        return new AffineMap(domains, symbols, results);
    }

    public static AffineMap FromCallable(Delegate func)
    {
        var parameters = func.Method.GetParameters();
        var arguments = new object[parameters.Length];
        var domains = new List<AffineDomain>();
        var symbols = new List<AffineSymbol>();
        for (int i = 0; i < arguments.Length; i++)
        {
            var type = parameters[i].ParameterType;
            if (type == typeof(AffineDomain))
            {
                var domain = F.Affine.Domain(i);
                domains.Add(domain);
                arguments[i] = domain;
            }
            else if (type == typeof(AffineSymbol))
            {
                var symbol = F.Affine.Symbol(symbols.Count);
                symbols.Add(symbol);
                arguments[i] = symbol;
            }
            else
            {
                throw new ArgumentException("Invalid callable argument");
            }
        }

        var results = (AffineRange[])func.DynamicInvoke(arguments)!;
        return new AffineMap(CollectionsMarshal.AsSpan(domains), CollectionsMarshal.AsSpan(symbols), results);
    }

    public static AffineMap Identity(int rank)
    {
        var domains = F.Affine.Domains(rank);
        var results = domains.Select(x => new AffineRange(x.Offset, x.Extent)).ToArray();
        return new AffineMap(domains, default, results);
    }

    public static AffineMap Permutation(int[] perms)
    {
        var domains = F.Affine.Domains(perms.Length);
        var results = Enumerable.Range(0, perms.Length).Select(i => new AffineRange(domains[perms[i]].Offset, domains[perms[i]].Extent)).ToArray();
        return new AffineMap(domains, default, results);
    }

    public bool IsProjectedPermutation(bool allowConstInResults)
    {
        if (Symbols.Length > 0)
        {
            return false;
        }

        // Having more results than inputs means that results have duplicated dims or
        // zeros that can't be mapped to input dims.
        if (Results.Length > Domains.Length && !allowConstInResults)
        {
            return false;
        }

        var seen = Enumerable.Repeat(false, Domains.Length).ToArray();
        foreach (var range in Results)
        {
            switch (range.Offset, range.Extent)
            {
                case (AffineDim dim, AffineExtent extent) when dim.Position == extent.Position:
                    if (seen[dim.Position])
                    {
                        return false;
                    }

                    seen[dim.Position] = true;
                    break;
                case (AffineConstant, AffineConstant):
                    if (!allowConstInResults)
                    {
                        return false;
                    }

                    break;
                default:
                    return false;
            }
        }

        return true;
    }

    public bool IsPermutation()
    {
        if (Domains.Length != Results.Length)
        {
            return false;
        }

        return IsProjectedPermutation(false);
    }

    public TIR.Range[] Apply(ReadOnlySpan<Expr> dims, ReadOnlySpan<Expr> extents, IReadOnlyDictionary<AffineSymbol, Expr>? symbols = null)
    {
        var newResults = new TIR.Range[Results.Length];
        for (int i = 0; i < newResults.Length; i++)
        {
            newResults[i] = Results[i].Apply(dims, extents, symbols);
        }

        return newResults;
    }

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitAffineMap(this, context);

    public AffineMap With(AffineDomain[]? domains = null, AffineSymbol[]? symbols = null, AffineRange[]? results = null)
        => new AffineMap(domains ?? Domains, symbols ?? Symbols, results ?? Results);

    public override string ToString()
    {
        var domains = string.Join(", ", Enumerable.Range(0, Domains.Length).Select(i => $"(d{i}, t{i})"));
        var syms = string.Join(", ", Enumerable.Range(0, Symbols.Length).Select(i => $"s{i}"));
        var results = StringUtility.Join(", ", Results.AsValueEnumerable().Select(expr => expr.GetDisplayString(Symbols)));

        return Symbols.Length == 0 ? $"({domains}) -> ({results})" : $"({domains})[{syms}] -> ({results})";
    }
}

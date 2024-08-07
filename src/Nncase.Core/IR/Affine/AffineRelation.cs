// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.Utilities;

namespace Nncase.IR.Affine;

public sealed class AffineRelation : Expr
{
    private readonly int _domainsCount;
    private readonly int _symbolsCount;

    public AffineRelation(ReadOnlySpan<AffineDim> domains, ReadOnlySpan<AffineSymbol> symbols, ReadOnlySpan<AffineExpr> results)
            : base(domains.ToArray().AsEnumerable<Expr>().Concat(symbols.ToArray()).Concat(results.ToArray()))
    {
        _domainsCount = domains.Length;
        _symbolsCount = symbols.Length;
    }

    public ReadOnlySpan<AffineDim> Domains => SpanUtility.UnsafeCast<Expr, AffineDim>(Operands.Slice(0, _domainsCount));

    public ReadOnlySpan<AffineSymbol> Symbols => SpanUtility.UnsafeCast<Expr, AffineSymbol>(Operands.Slice(_domainsCount, _symbolsCount));

    public ReadOnlySpan<AffineExpr> Results => SpanUtility.UnsafeCast<Expr, AffineExpr>(Operands.Slice(_domainsCount + _symbolsCount));

    public static AffineRelation operator *(AffineRelation lhs, AffineRelation rhs)
    {
        if (lhs.Results.Length != rhs.Domains.Length)
        {
            throw new ArgumentException("Cannot compose AffineMaps with mismatching dimensions and results.");
        }

        var results = rhs.Results.AsValueEnumerable().Select(x => x.ReplaceDomainsAndSymbols(lhs.Results.AsValueEnumerable().Select(r => new AffineRange(r, 0)).ToArray(), Array.Empty<AffineSymbol>())).ToArray();
        var symbols = lhs.Symbols.ToArray().Concat(rhs.Symbols.ToArray()).ToArray();
        return new AffineRelation(lhs.Domains, symbols, results);
    }

    public static AffineRelation FromCallable(Func<AffineDim[], AffineSymbol[], AffineExpr[]> func, int dimsCount, int symbolsCount = 0)
    {
        var domains = F.Affine.Dims(dimsCount);
        var symbols = F.Affine.Symbols(symbolsCount);
        var results = func(domains, symbols);
        return new AffineRelation(domains, symbols, results);
    }

    public static AffineRelation FromCallable(Delegate func)
    {
        var parameters = func.Method.GetParameters();
        var arguments = new object[parameters.Length];
        var domains = new List<AffineDim>();
        var symbols = new List<AffineSymbol>();
        for (int i = 0; i < arguments.Length; i++)
        {
            var type = parameters[i].ParameterType;
            if (type == typeof(AffineDim))
            {
                var domain = F.Affine.Dim(i);
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

        var results = (AffineExpr[])func.DynamicInvoke(arguments)!;
        return new AffineRelation(CollectionsMarshal.AsSpan(domains), CollectionsMarshal.AsSpan(symbols), results);
    }

    public static AffineRelation Identity(int rank)
    {
        var domains = F.Affine.Dims(rank);
        var results = domains.Select(x => x).ToArray();
        return new AffineRelation(domains, default, results);
    }

    public Expr Apply(ReadOnlySpan<Expr> dims, IReadOnlyDictionary<AffineSymbol, Expr>? symbols = null)
    {
        var newResults = new Expr[Results.Length];
        for (int i = 0; i < newResults.Length; i++)
        {
            newResults[i] = Results[i].Apply(dims, Array.Empty<Expr>(), symbols);
        }

        return newResults;
    }

    public AffineRelation Inverse()
    {
        var domains = new AffineDim[Results.Length];
        var ranges = new AffineExpr[Domains.Length];
        var syms = new List<AffineSymbol>();
        for (int i = 0; i < Results.Length; i++)
        {
            domains[i] = new AffineDim(i);
            var offset = AffineUtility.Inverse<AffineDim>(Results[i], domains[i], out var independentDimVar);
            switch (independentDimVar)
            {
                case AffineDim dim:
                    ranges[dim.Position] = offset;
                    break;
                default:
                    throw new System.Diagnostics.UnreachableException();
            }
        }

        for (int i = 0; i < ranges.Length; i++)
        {
            if (ranges[i] is null)
            {
                var s = new AffineSymbol(i);
                syms.Add(s);
                ranges[i] = s;
            }
        }

        return new(domains, syms.ToArray(), ranges);
    }

    public override TExprResult Accept<TExprResult, TTypeResult, TContext>(ExprFunctor<TExprResult, TTypeResult, TContext> functor, TContext context) => functor.VisitAffineRelation(this, context);

    public override string ToString()
    {
        var domains = string.Join(", ", Enumerable.Range(0, Domains.Length).Select(i => $"d{i}"));
        var syms = string.Join(", ", Enumerable.Range(0, Symbols.Length).Select(i => $"s{i}"));
        var results = StringUtility.Join(", ", Results.AsValueEnumerable().Select(expr => expr.GetDisplayString(Symbols)));

        return Symbols.Length == 0 ? $"({domains}) -> ({results})" : $"({domains})[{syms}] -> ({results})";
    }

    public AffineRelation With(AffineDim[]? domains = null, AffineSymbol[]? symbols = null, AffineExpr[]? results = null)
        => new AffineRelation(domains ?? Domains, symbols ?? Symbols, results ?? Results);
}

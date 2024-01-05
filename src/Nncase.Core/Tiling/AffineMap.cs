// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Tensors;

namespace Nncase.Tiling;

public sealed class AffineMap
{
    public AffineMap(int dimsCount, AffineSymbolExpr[] symbols, AffineExpr[] results)
    {
        DimsCount = dimsCount;
        Symbols = symbols;
        Results = results;
    }

    public int DimsCount { get; }

    public AffineSymbolExpr[] Symbols { get; }

    public AffineExpr[] Results { get; }

    public static AffineMap operator *(AffineMap lhs, AffineMap rhs)
    {
        if (lhs.Results.Length != rhs.DimsCount)
        {
            throw new ArgumentException("Cannot compose AffineMaps with mismatching dimensions and results.");
        }

        var results = rhs.Results.Select(x => x.ReplaceDims(lhs.Results)).ToArray();
        var symbols = lhs.Symbols.Concat(rhs.Symbols).ToArray();
        return new AffineMap(lhs.DimsCount, symbols, results);
    }

    public static AffineMap FromCallable(Func<IReadOnlyList<AffineDimExpr>, IReadOnlyList<AffineSymbolExpr>, AffineExpr[]> func, int dimsCount, int symbolsCount = 0)
    {
        var dims = Affine.Dims(dimsCount);
        var symbols = Affine.Symbols(symbolsCount);
        var results = func(dims, symbols);
        return new AffineMap(dimsCount, symbols, results);
    }

    public Expr[] Apply(IReadOnlyList<Expr> dims, IReadOnlyDictionary<AffineSymbolExpr, Expr> symbols) =>
        Results.Select(x => x.Apply(dims, symbols)).ToArray();

    public override string ToString()
    {
        var dims = string.Join(", ", Enumerable.Range(0, DimsCount).Select(i => $"d{i}"));
        var syms = string.Join(", ", Enumerable.Range(0, Symbols.Length).Select(i => $"s{i}"));
        var results = string.Join(", ", Results.Select(expr => expr.GetDisplayString(Symbols)));

        return Symbols.Length == 0 ? $"({dims}) -> ({results})" : $"({dims})[{syms}] -> ({results})";
    }
}

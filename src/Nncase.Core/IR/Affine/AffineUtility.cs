// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR.Affine;

public static class AffineUtility
{
    public static AffineExpr Inverse<T>(AffineExpr original, T input, out T? independentVar)
            where T : AffineExpr
    {
        var collector = new AffineInverseCollector();
        collector.Visit(original, default);
        var inverser = new AffineInverser<T>(collector.ExprMemo);
        var output = inverser.Visit(original, input);
        independentVar = inverser.IndependentVariable;
        return output;
    }

    public static AffineMap Inverse(AffineMap map, params long[] bounds)
    {
        var domains = new AffineDomain[map.Results.Length];
        var ranges = new AffineRange[map.Domains.Length];
        var syms = new List<AffineSymbol>();
        for (int i = 0; i < map.Results.Length; i++)
        {
            domains[i] = new(new AffineDim(i), new AffineExtent(i));
            var offset = Inverse(map.Results[i].Offset, domains[i].Offset, out var independentDimVar);
            var extent = Inverse(map.Results[i].Extent, domains[i].Extent, out var independentExtVar);
            switch (independentDimVar, independentExtVar)
            {
                case (AffineDim dim, AffineExtent ext) when dim.Position == ext.Position:
                    ranges[dim.Position] = new(offset, extent);
                    break;
                default:
                    throw new System.Diagnostics.UnreachableException();
            }
        }

        for (int i = 0; i < ranges.Length; i++)
        {
            if (ranges[i] is null)
            {
                ranges[i] = new(0, bounds[i]);
            }
        }

        return new(domains, syms.ToArray(), ranges);
    }

#if false
    private sealed class AutoTileRewriter : ExprRewriter
    {
        protected override Expr RewriteLeafFor(For expr)
        {
            if (expr.MemoryLevel == 0 || expr.Body is For)
            {
                return expr;
            }
            else
            {
                var levels = expr.MemoryLevel + 1;
                var outputRank = expr.CheckedShape.Rank;
                var domains = new AffineMap[levels, outputRank];
                var lastDomain = expr.Domain;
                for (int level = expr.MemoryLevel; level >= 0; level--)
                {
                    for (int j = 0; j < outputRank; j++)
                    {
                        var tileSizes = F.Affine.Symbols(outputRank);
                        var domain = new AffineMap(lastDomain.)
                    }
                }
                var childFor = expr.With(memoryLevel: 0);
            }
        }
    }
#endif
}

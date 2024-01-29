// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.IR.Affine;

public sealed record class TiledFor(For For, AffineSymbol[] TileSizes);

public static class AffineUtility
{

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

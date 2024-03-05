// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.IR.Affine;

namespace Nncase.Passes.Transforms;

public sealed class AutoTilePass : ModulePass
{
    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        var funcs = input.Functions.Count;
        for (int i = 0; i < funcs; i++)
        {
            var rewriter = new AutoTileRewriter(input);
            input.Replace(i, (BaseFunction)rewriter.Rewrite(input.Functions[i]));
        }

        return Task.FromResult(input);
    }

    private sealed class AutoTileRewriter : ExprRewriter
    {
        private readonly IRModule _module;

        public AutoTileRewriter(IRModule module)
        {
            _module = module;
        }

        protected override Expr RewriteLeafGrid(Grid grid)
        {
            var call = CompilerServices.Tile(grid, _module);
            return call;
        }
    }
}

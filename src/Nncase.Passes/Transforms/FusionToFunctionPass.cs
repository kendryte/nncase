// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Nncase.IR;

namespace Nncase.Passes.Transforms;

public sealed class FusionToFunctionPass : ModulePass
{
    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        var count = input.Functions.Count;
        for (int i = 0; i < count; i++)
        {
            var replacer = new FusionReplacer(input);
            input.Replace(i, (BaseFunction)replacer.Rewrite(input.Functions[i]));
        }

        return Task.FromResult(input);
    }

    private sealed class FusionReplacer : ExprRewriter
    {
        private readonly IRModule _module;

        public FusionReplacer(IRModule module)
        {
            _module = module;
        }

        protected override Expr RewriteLeafFusion(Fusion expr)
        {
            var func = new Function(expr.Body, expr.Parameters, expr.ModuleKind);
            _module.Add(func);
            return func;
        }
    }
}

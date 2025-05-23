// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Threading.Tasks;
using System.Xml;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Math;
using Nncase.IR.Tensors;
using Nncase.PatternMatch;
using Nncase.Targets;
using static Nncase.PatternMatch.F.Math;
using static Nncase.PatternMatch.Utility;
using static Nncase.Utilities.ReplaceUtility;

namespace Nncase.Passes.Rules.ShapeBucket;

public sealed class RemoveUnusedFunctions : ModulePass
{
    public RemoveUnusedFunctions(CompileOptions compileOptions)
    {
        CompileOptions = compileOptions;
    }

    public CompileOptions CompileOptions { get; }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        while (true)
        {
            var funcs = new HashSet<BaseFunction>(ReferenceEqualityComparer.Instance);
            if (input.Entry is not null)
            {
                funcs.Add(input.Entry);
                var collector = new FuncCollector(funcs);
                collector.Visit(input.Entry);
            }

            var toRemove = input.Functions.Except(funcs).ToArray();
            if (toRemove.Length == 0)
            {
                break;
            }

            foreach (var funcToRemove in toRemove)
            {
                input.Remove(funcToRemove);
            }
        }

        return Task.FromResult(input);
    }

    private class FuncCollector : ExprWalker
    {
        public FuncCollector(HashSet<BaseFunction> funcs)
            : base(true)
        {
            Funcs = funcs;
        }

        public HashSet<BaseFunction> Funcs { get; }

        protected override Unit VisitLeafBaseFunction(BaseFunction expr)
        {
            Funcs.Add(expr);
            return default;
        }
    }
}

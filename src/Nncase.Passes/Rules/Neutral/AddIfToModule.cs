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

public sealed class AddFunctionToModule : ModulePass
{
    public AddFunctionToModule(CompileOptions compileOptions)
    {
        CompileOptions = compileOptions;
    }

    public CompileOptions CompileOptions { get; }

    protected override Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        while (true)
        {
            var funcs = new HashSet<BaseFunction>(ReferenceEqualityComparer.Instance);
            foreach (var func in input.Functions)
            {
                var collector = new FuncCollector(funcs);
                collector.Visit(func);
            }

            var toAdd = funcs.Except(input.Functions).ToArray();
            if (toAdd.Length == 0)
            {
                break;
            }

            foreach (var ifToAdd in toAdd)
            {
                input.Add(ifToAdd);
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

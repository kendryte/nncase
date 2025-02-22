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
            var funcsToRemove = new HashSet<BaseFunction>(ReferenceEqualityComparer.Instance);
            foreach (var func in input.Functions)
            {
                IRHelpers.DCE(func);
                if (!ReferenceEquals(func, input.Entry)
                    && func.Users.Count == 1)
                {
                    funcsToRemove.Add(func);
                }
            }

            if (funcsToRemove.Count == 0)
            {
                break;
            }

            foreach (var func in funcsToRemove)
            {
                input.Remove(func);
            }
        }

        return Task.FromResult(input);
    }
}

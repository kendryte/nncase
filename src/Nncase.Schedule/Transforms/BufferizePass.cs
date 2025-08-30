// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes.Rules.ShapeBucket;
using Nncase.Schedule;
using Nncase.Schedule.Bufferize;
using Nncase.TIR;
using Nncase.Utilities;

namespace Nncase.Passes.Transforms;

public sealed class BufferizePass : ModulePass
{
    protected override async Task<IRModule> RunCoreAsync(IRModule input, RunPassContext context)
    {
        foreach (var funcs in input.Functions.OfType<PrimFunction>().GroupBy(x => x.ModuleKind))
        {
            var bufferizeVisitor = new BufferizeVisitor(funcs);
            bufferizeVisitor.Bufferize();
        }

        await new AddFunctionToModule(CompileSession.CompileOptions).RunAsync(input, context);
        await new RemoveUnusedFunctions(CompileSession.CompileOptions).RunAsync(input, context);
        return input;
    }
}

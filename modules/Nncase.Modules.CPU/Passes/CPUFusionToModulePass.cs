// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes.Analysis;
using Nncase.Passes.Mutators;
using Nncase.Passes.Tile;
using Nncase.Targets;
using Nncase.TIR;

namespace Nncase.Passes;

internal sealed class CPUFusionToModulePass : ModulePass
{
    /// <inheritdoc/>
    protected override Task<IRModule> RunCoreAsync(IRModule module, RunPassContext options)
    {
        foreach (var item in ExprCollector.Collect(module.Entry!).OfType<Fusion>().Where(f => f.ModuleKind == CPUTarget.Kind))
        {
            module.Add(item);
        }

        return Task.FromResult(module);
    }
}

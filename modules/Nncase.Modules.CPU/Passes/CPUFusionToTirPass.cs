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

internal sealed class CPUFusionToTirPass : ModulePass
{
    private readonly TileOptions _tileOptions;

    public CPUFusionToTirPass(TileOptions tileOptions)
    {
        _tileOptions = tileOptions;
    }

    private IAnalyzerManager AnalyzerManager => CompileSession.GetRequiredService<IAnalyzerManager>();

    /// <inheritdoc/>
    protected override Task<IRModule> RunCoreAsync(IRModule module, RunPassContext options)
    {
        Dictionary<Fusion, BaseFunction> fusionConertedCache = new(ReferenceEqualityComparer.Instance);

        for (int i = 0; i < module.Functions.Count; i++)
        {
            if (module.Functions[i] is Function { ModuleKind: string kind } func && kind == Callable.StackVMModuleKind)
            {
                var analysis = new Dictionary<Type, IAnalysisResult> { [typeof(IExprUserAnalysisResult)] = AnalyzerManager.GetAnaylsis<IExprUserAnalysisResult>(func), };
                var rewriter = new DataFlowMergeRewriter();
                var fusionCheckCache = new Dictionary<Fusion, IFusionChecker>(ReferenceEqualityComparer.Instance);

                var post = (Function)rewriter.Rewrite(
                    func,
                    new Mutators.IMergeRewriteRule[] { new CPUSameInputFusionMergeRule() },
                    (rule, option) => new CPUFusionGroupMutator<MultiFusionChecker>(fusionCheckCache, _tileOptions, rule, option),
                    new() { AnalysisResults = analysis, MatchOptions = new Mutators.FusionGroupMutator.GroupedMatchOptions() });

                var mutator = new CheckedConvertMutator(fusionConertedCache, fusionCheckCache, _tileOptions, options);
                var new_func = (Function)mutator.Rewrite(post);
                CompilerServices.InferenceType(new_func);
                if (mutator.IsMutated)
                {
                    module.Replace(i, new_func);
                }
            }
        }

        foreach (var item in fusionConertedCache.Values)
        {
            if (item is PrimFunctionWrapper wrapper)
            {
                module.Add(wrapper);
                module.Add(wrapper.Target);
            }
        }

        return Task.FromResult(module);
    }
}

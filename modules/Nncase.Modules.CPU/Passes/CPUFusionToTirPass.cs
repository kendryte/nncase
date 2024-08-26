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
    private readonly CompileOptions _compileOptions;

    public CPUFusionToTirPass(CompileOptions compileOptions)
    {
        _compileOptions = compileOptions;
    }

    /// <inheritdoc/>
    protected override Task<IRModule> RunCoreAsync(IRModule module, RunPassContext options)
    {
        HashSet<PrimFunctionWrapper> kernelFuncs = new(ReferenceEqualityComparer.Instance);
        HashSet<PrimFunction> deviceFuncs = new(ReferenceEqualityComparer.Instance);

        for (int i = 0; i < module.Functions.Count; i++)
        {
            if (module.Functions[i] is Fusion { ModuleKind: CPUTarget.Kind } fusion)
            {
                // var analysis = new Dictionary<Type, IAnalysisResult>
                // {
                //     [typeof(IExprUserAnalysisResult)] = AnalyzerManager.GetAnaylsis<IExprUserAnalysisResult>(module.Functions[i]),
                // };
                // var rewriter = new DataFlowMergeRewriter();
                var fusionCheckCache = new Dictionary<Fusion, FusionChecker>(ReferenceEqualityComparer.Instance);

                // var post = (Fusion)rewriter.Rewrite(
                //     fusion,
                //     new IMergeRewriteRule[] {
                //       new CPUSameInputFusionMergeRule(),
                //       new CPUMultiInputFusionMergeRule(),
                //     },
                //     (rule, option) => new CPUFusionGroupMutator(fusionCheckCache, rule, option),
                //     new() { AnalysisResults = analysis, MatchOptions = new FusionGroupMutator.GroupedMatchOptions() });
                // if (DumpScope.Current.IsEnabled(DumpFlags.PassIR))
                // {
                //     DumpScope.Current.DumpIR(post, string.Empty, "L2Tiled");
                // }
                var post = fusion;
                var primBody = new List<Expr>();
                var visitor = new KernelToTIRVisitor(primBody, deviceFuncs, fusionCheckCache, new BufferSchedule.BufferScheduler(_compileOptions.TargetOptions is null ? new CpuTargetOptions().HierarchySizes[0] : ((CpuTargetOptions)_compileOptions.TargetOptions).HierarchySizes[0]), new BufferSchedule.LifeTimeCollector());
                visitor.Convert(post);
                var primFunc = T.PrimFunc(post.Name, post.ModuleKind, visitor.InputBuffers.Concat(visitor.OutputBuffers).ToArray()).Body(primBody.ToArray()).Build();
                primFunc.SchedResult.DataUsage = visitor.DataUsage;
                primFunc.SchedResult.DataAlign = visitor.MaxDTypeSize;
                var primWrapper = new PrimFunctionWrapper(primFunc, visitor.InputBuffers.Count());
                module.Replace(i, primWrapper);
                kernelFuncs.Add(primWrapper);
            }
        }

        foreach (var item in kernelFuncs)
        {
            module.Add(item.Target);
        }

        foreach (var item in deviceFuncs)
        {
            module.Add(item);
        }

        return Task.FromResult(module);
    }
}

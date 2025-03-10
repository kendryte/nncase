// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.Passes.Analysis;
using Nncase.Passes.Mutators;
using Nncase.Passes.Tile;
using Nncase.Targets;
using Nncase.TIR;

namespace Nncase.Passes;

/// <summary>
/// BufferSizeCalculator for cpu.
/// now it supports to optimize memory move for:
///    1. boxing store.
///    2. reshape.
/// </summary>
public sealed class CpuBufferSizeCalculator : BufferSchedule.BufferSizeCalculator
{
    protected override Result VisitCall(Call expr)
    {
        if (expr is Call { Target: IR.Distributed.Boxing boxing } && boxing.NewType is TensorType ttype)
        {
            var res = VisitType(ttype);
            return new(0, res.Shape, res.Stride);
        }
        else if (expr.Target is IR.Tensors.Reshape)
        {
            var res = VisitType(expr.CheckedType);
            return new(0, res.Shape, res.Stride);
        }

        return base.VisitCall(expr);
    }
}

public sealed class CpuLifeTimeUpdater : BufferSchedule.LifeTimeUpdater
{
    protected override Unit VisitCall(Call expr, Context context)
    {
        foreach (var item in expr.Arguments)
        {
            if (item is IR.Tuple tp)
            {
                Visit(tp, context);
            }
            else if (item is Call c)
            {
                if (c.Target is IR.Tensors.GetItem)
                {
                    Visit(c, context);
                }
                else if (c.Target is IR.Tensors.Reshape)
                {
                    Visit(c, context);
                }
                else
                {
                    PerformUpdate(c, context);
                }
            }
        }

        PerformUpdate(expr, context);
        return default;
    }
}

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
                var memCapacity = _compileOptions.TargetOptions is CpuTargetOptions copt ? copt.HierarchySizes[0] : throw new ArgumentException("TargetOptions is not CpuTargetOptions");
                var visitor = new KernelToTIRVisitor(primBody, deviceFuncs, fusionCheckCache, new BufferSchedule.BufferScheduler(memCapacity), new BufferSchedule.LifeTimeCollector(new CpuLifeTimeUpdater(), new CpuBufferSizeCalculator()));
                visitor.Convert(post);
                var dimVars = visitor.DimVars.ToArray();
                var primFunc = T.PrimFunc(post.Name, post.ModuleKind, dimVars.Cast<Expr>().Concat(visitor.InputBuffers).Concat(visitor.OutputBuffers).ToArray()).Body(primBody.ToArray()).Build();
                primFunc.SchedResult.DataUsage = visitor.DataUsage;
                primFunc.SchedResult.DataAlign = Math.Max(8, visitor.MaxDTypeSize);
                var primWrapper = new PrimFunctionWrapper(primFunc, dimVars.Length + visitor.InputBuffers.Count());
                module.Replace(i, primWrapper);
                kernelFuncs.Add(primWrapper);

                AddDimVarsToWrapperCallers(primWrapper, dimVars, visitor.OutputBuffers);
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

    private void AddDimVarsToWrapperCallers(PrimFunctionWrapper primWrapper, IEnumerable<Var> dimVars, IEnumerable<TIR.Buffer> outputBuffers)
    {
        var callers = primWrapper.Users.OfType<Call>().ToArray();
        var outputBufferShapes = outputBuffers.Select(x => (x.ElemType, Shape: new Shape(x.Dimensions).ToValueArrayExpr())).ToArray();
        foreach (var caller in callers)
        {
            var outputAllocs = outputBufferShapes.Select(x => IR.F.Buffer.Uninitialized(x.ElemType, TIR.MemoryLocation.Data, x.Shape));
            var newArgs = dimVars.Cast<Expr>().Concat(caller.Arguments.ToArray()).Concat(outputAllocs).ToArray();
            var newCaller = caller.With(arguments: newArgs);
            IRHelpers.ReplaceAllUsesWith(caller, newCaller);
        }
    }
}

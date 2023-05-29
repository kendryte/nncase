// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics.CodeAnalysis;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Extensions.DependencyInjection;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.Passes.Analysis;
using Nncase.Passes.Rules;
using Nncase.PatternMatch;
using Nncase.TIR;
using static Nncase.PatternMatch.Utility;

namespace Nncase.Passes;

/// <summary>
/// merge call/ assgin ddr buffer start/layout.
/// </summary>
public sealed class DDrBufferSchdeulePass : ModulePass
{
    private readonly Dictionary<string, Dictionary<Schedule.MemoryLocation, int>> _module_usage = new();

    private readonly Dictionary<string, HashSet<TIR.Buffer>> _module_hashset = new();

    private readonly bool _enbaleMergeCall;

    private IAnalyzerManager AnalyzerManager => CompileSession.GetRequiredService<IAnalyzerManager>();

    public DDrBufferSchdeulePass(bool enableMergeCall = false)
    {
        _enbaleMergeCall = enableMergeCall;
    }

    /// <inheritdoc/>
    protected override async Task<IRModule> RunCoreAsync(IRModule module, RunPassContext options)
    {
        // 1. merge the all call prim func
        if (_enbaleMergeCall)
        {
            HashSet<BaseFunction> mergedFuncs = new(ReferenceEqualityComparer.Instance);
            HashSet<BaseFunction> stackvmFuncs = new(ReferenceEqualityComparer.Instance);
            for (int i = 0; i < module.Functions.Count; i++)
            {
                if (module.Functions[i] is Function { ModuleKind: "stackvm" } func)
                {
                    var analysis = new Dictionary<Type, IAnalysisResult>
                    {
                        [typeof(IExprUserAnalysisResult)] = AnalyzerManager.GetAnaylsis<IExprUserAnalysisResult>(func),
                    };
                    _ = new HashSet<BaseFunction>(ReferenceEqualityComparer.Instance);
                    var mergePass = new DataflowPass();
                    mergePass.Add<Rules.Neutral.PrimFuncMergeRule>(mergedFuncs);
                    var post = await mergePass.RunAsync(func, new() { AnalysisResults = analysis, RewriteOnce = true });
                    module.Replace(i, post);
                    stackvmFuncs.Add(post);
                }
            }

            // 2. add the ext func into module.
            foreach (var func in stackvmFuncs)
            {
                var collector = new ExternalFuncCollector();
                collector.Visit(func);
                foreach (var ext_func in collector.GetExternalFuncs())
                {
                    module.Add(ext_func);
                }
            }

            // 3. remove the all merged funcs
            foreach (var item in mergedFuncs)
            {
                module.Remove(item);
            }
        }

        // 4. schedule the prim funcs.
        for (int i = 0; i < module.Functions.Count; i++)
        {
            if (module.Functions[i] is TIR.PrimFunction prim_func)
            {
                if (!prim_func.SchedResult.IsScheduled)
                {
                    var ddr_allocator = new DDrBufferAllocator(_module_usage, _module_hashset);
                    ddr_allocator.Visit(prim_func); // changed ddr buffer.
                    prim_func.SchedResult.DataUsage = ddr_allocator.DataUsage;
                    prim_func.SchedResult.IsScheduled = ddr_allocator.Changed;
                }
            }
        }

        _module_hashset.Clear();
        _module_usage.Clear();

        return await Task.FromResult(module);
    }
}

/// <summary>
/// collect and assgin the PhysicalBuffer.
/// </summary>
internal sealed class DDrBufferAllocator : ExprVisitor<bool, bool>
{
    private readonly Dictionary<Schedule.MemoryLocation, int> _functionUsage;
    private readonly HashSet<TIR.Buffer> _functionHashset;

    private PrimFunction? _entry;

    public DDrBufferAllocator(Dictionary<string, Dictionary<Schedule.MemoryLocation, int>> module_usage, Dictionary<string, HashSet<TIR.Buffer>> module_hashset)
    {
        ModuleUsage = module_usage;
        ModuleHashSet = module_hashset;
        _functionUsage = new();
        _functionHashset = new(ReferenceEqualityComparer.Instance);
        Changed = false;
    }

    public Dictionary<string, Dictionary<Schedule.MemoryLocation, int>> ModuleUsage { get; }

    public Dictionary<string, HashSet<TIR.Buffer>> ModuleHashSet { get; }

    public bool Changed { get; private set; }

    public int DataUsage => _functionUsage.GetValueOrDefault(Schedule.MemoryLocation.Data, 0);

    /// <remarks>
    /// only visit one prim func.
    /// </remarks>
    protected override bool VisitPrimFunction(PrimFunction primFunction)
    {
        _entry ??= primFunction;
        if (object.ReferenceEquals(_entry, primFunction))
        {
            foreach (var physical in primFunction.Parameters)
            {
                if (physical.MemLocation is Schedule.MemoryLocation.Input or Schedule.MemoryLocation.Output)
                {
                    // avoid visit same buffer
                    if (!_functionHashset.Contains(physical))
                    {
                        // input/output write into the FunctionUsage
                        if (!_functionUsage.TryGetValue(physical.MemLocation, out var start))
                        {
                            start = 0;
                        }

                        physical.Start = start;
                        _functionUsage[physical.MemLocation] = start + physical.Size;
                        _functionHashset.Add(physical);
                        Changed = true;
                    }
                }
                else
                {
                    throw new NotSupportedException($"The prim function parameters mem location must be input/output but get {physical.MemLocation}!");
                }
            }

            return base.VisitPrimFunction(_entry);
        }

        return true;
    }

    protected override bool VisitLeafBuffer(TIR.Buffer buffer)
    {
        if (buffer is not TIR.PhysicalBuffer physical)
        {
            return true;
        }

        // rdata write into the moduleUsage
        if (physical.MemLocation is Schedule.MemoryLocation.Rdata)
        {
            if (!ModuleHashSet.TryGetValue(_entry!.ModuleKind, out var module_hashset))
            {
                module_hashset = new(ReferenceEqualityComparer.Instance);
                ModuleHashSet.Add(_entry!.ModuleKind, module_hashset);
            }

            if (!ModuleUsage.TryGetValue(_entry!.ModuleKind, out var module_usage))
            {
                module_usage = new();
                ModuleUsage.Add(_entry!.ModuleKind, module_usage);
            }

            if (!module_hashset.Contains(physical))
            {
                if (!module_usage.TryGetValue(physical.MemLocation, out var start))
                {
                    start = 0;
                }

                physical.Start = start;
                module_usage[physical.MemLocation] = start + physical.Size;
                module_hashset.Add(physical);
                _entry.SchedResult.Rdatas.Add(physical);

                Changed = true;
            }
        }
        else if (physical.MemLocation is Schedule.MemoryLocation.Data)
        {
            // data write into the FunctionUsage
            if (!_functionHashset.Contains(physical))
            {
                if (!_functionUsage.TryGetValue(physical.MemLocation, out var start))
                {
                    start = 0;
                }

                physical.Start = start;
                _functionUsage[physical.MemLocation] = start + physical.Size;
                _functionHashset.Add(physical);
                Changed = true;
            }
        }
        else if (physical.MemLocation is Schedule.MemoryLocation.SharedData)
        {
            throw new NotSupportedException("Current Not Support!");
        }

        return true;
    }

    protected override bool DefaultVisitLeaf(Expr expr) => true;
}

internal sealed class ExternalFuncCollector : ExprWalker
{
    public HashSet<BaseFunction> GetExternalFuncs()
    {
        var set = new HashSet<BaseFunction>(ReferenceEqualityComparer.Instance);
        set.UnionWith(ExprMemo.Keys.OfType<PrimFunctionWrapper>());
        set.UnionWith(set.OfType<PrimFunctionWrapper>().Select(w => w.Target).ToArray());
        return set;
    }
}

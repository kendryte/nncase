// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.Passes.Rules;
using Nncase.TIR;

namespace Nncase.Passes;

/// <summary>
/// merge call/ assgin ddr buffer start/layout.
/// </summary>
public sealed class DDrBufferSchdeulePass : ModulePass
{
    private readonly Dictionary<string, Dictionary<Schedule.MemoryLocation, int>> _module_usage = new();
    private readonly Dictionary<string, HashSet<TIR.Buffer>> _module_hashset = new();

    /// <inheritdoc/>
    protected override Task<IRModule> RunCoreAsync(IRModule module, RunPassContext options)
    {
        for (int i = 0; i < module.Functions.Count; i++)
        {
            if (module.Functions[i] is TIR.PrimFunction prim_func)
            {
                if (!prim_func.SchedResult.IsScheduled)
                {
                    var ddr_allocator = new DDrBufferAllocator(_module_usage, _module_hashset);
                    ddr_allocator.Visit(prim_func); // changed ddr buffer.
                    prim_func.SchedResult.IsScheduled = ddr_allocator.Changed;
                }
            }
        }

        return Task.FromResult(module);
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
        else if (physical.MemLocation is Schedule.MemoryLocation.Data or Schedule.MemoryLocation.SharedData)
        {
            throw new NotSupportedException("Current Not Support!");
        }

        return true;
    }

    protected override bool DefaultVisitLeaf(Expr expr) => true;
}

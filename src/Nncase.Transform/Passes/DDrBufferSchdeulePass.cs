// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;
using Nncase.TIR;
using Nncase.Transform.Rules;

namespace Nncase.Transform.Passes;

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
    public readonly Dictionary<string, Dictionary<Schedule.MemoryLocation, int>> ModuleUsage;
    public readonly Dictionary<string, HashSet<TIR.Buffer>> ModuleHashSet;
    public bool Changed;
    public PrimFunction? _entry;
    private readonly Dictionary<Schedule.MemoryLocation, int> _functionUsage;
    private readonly HashSet<TIR.Buffer> _functionHashset;

    public DDrBufferAllocator(Dictionary<string, Dictionary<Schedule.MemoryLocation, int>> module_usage, Dictionary<string, HashSet<TIR.Buffer>> module_hashset)
    {
        ModuleUsage = module_usage;
        ModuleHashSet = module_hashset;
        _functionUsage = new();
        _functionHashset = new(ReferenceEqualityComparer.Instance);
        Changed = false;
    }

    /// <remarks>
    /// only visit one prim func.
    /// </remarks>
    /// <returns></returns>
    public override bool Visit(PrimFunction primFunction)
    {
        _entry ??= primFunction;
        if (object.ReferenceEquals(_entry, primFunction))
        {
            return base.Visit(_entry);
        }

        return true;
    }

    public override bool VisitLeaf(TIR.Buffer buffer)
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
        else if (physical.MemLocation is Schedule.MemoryLocation.Input or Schedule.MemoryLocation.Output)
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
        else if (physical.MemLocation is Schedule.MemoryLocation.Data or Schedule.MemoryLocation.SharedData)
        {
            throw new NotSupportedException("Current Not Support!");
        }

        return true;
    }

    public override bool DefaultVisitLeaf(Expr expr) => true;

    public override object DefaultVisitLeaf(IVisitable visitable) => true;
}

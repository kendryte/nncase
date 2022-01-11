// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
namespace Nncase.Schedule;

using AllocationMap = Dictionary<IR.Expr, BufferAllocation>;


/// <summary>
/// the memory type
/// </summary>
public enum MemoryLocation : byte
{
    Input = 0,
    Output = 1,
    Rdata = 2,
    Data = 3,
    SharedData = 4,
    PrivateBase = 64,
}

/// <summary>
/// memory range define.
/// </summary>
public struct MemoryRange
{
    public MemoryLocation MemoryLocate;
    public DataType DType;
    public UInt16 SharedModule;
    public uint Start;
    public uint Size;
}

public struct BufferAllocation
{
    public MemoryLocation MemoryLocate;
    public DataType DType;
    public ulong SharedModule;
    public ulong Start;
    public ulong Size;
    public IR.Shape Shape;
    public IR.Shape Strides;
    public IR.Shape StridesShape;

    public ulong LinearEnd() => Start + Size;

    public bool Overlap(BufferAllocation rhs) => Size != 0 && rhs.Size != 0 && this.MemoryLocate == rhs.MemoryLocate && (Start < rhs.LinearEnd() && LinearEnd() > rhs.Start);

    public MemoryRange RuntimeType => new()
    {
        MemoryLocate = this.MemoryLocate,
        DType = this.DType,
        SharedModule = (UInt16)SharedModule,
        Start = (uint)this.Start,
        Size = (uint)this.Size
    };
}

public struct SchedModelResult
{
    /// <summary>
    /// sched module result
    /// </summary>
    public List<SchedModuleResult> Modules;

    /// <summary>
    /// sched function result
    /// </summary>
    public SchedFunctionResult Entry;

    /// <summary>
    /// the parent ir module
    /// </summary>
    public IR.IRModule ParentModule;
}

public struct SchedModuleResult
{
    /// <summary>
    /// current Module type
    /// </summary>
    public CodeGen.ModuleType ModuleType;

    /// <summary>
    /// contains functions
    /// </summary>
    public List<SchedFunctionResult> Functions;

    /// <summary>
    /// schedfunction maps
    /// </summary>
    public Dictionary<IR.Expr, SchedFunctionResult> FunctionsMap;

    /// <summary>
    /// the buffer allocations
    /// </summary>
    public AllocationMap Allocations;

    /// <summary>
    /// mem collection
    /// </summary>
    public Dictionary<MemoryLocation, ulong> MaxUsages;

    /// <summary>
    /// shared mem
    /// </summary>
    public Dictionary<CodeGen.ModuleType, ulong> SharedMaxUsages;
}

public struct SchedFunctionResult
{
    public IR.IRModule ParentModule;
    public SchedModuleResult SchedModule;
    public ulong InputPoolSize;
    public ulong OutputPoolSize;
    public List<IR.Expr> ComputeSequence;
    public IR.Function Function;
}

/// <summary>
/// the scheduler interface
/// </summary>
public interface IScheduler
{
    /// <summary>
    /// the current target
    /// </summary>
    public ITarget Target { get; set; }
    /// <summary>
    /// the main module
    /// </summary>
    public IR.IRModule ParentModule { get; set; }

    /// <summary>
    /// multi stage schedules.
    /// relay IR -> TIR -> lowered TIR 
    /// </summary>
    /// <param name="skip_buffer_alias"></param>
    /// <returns></returns>
    public SchedModelResult Schedule(bool skip_buffer_alias = false);
}
// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
namespace Nncase.Schedule;

using AllocationMap = Dictionary<IR.Expr, BufferAllocation>;

/// <summary>
/// the memory type.
/// </summary>
public enum MemoryLocation : byte
{
    /// <summary>
    /// input.
    /// </summary>
    Input = 0,

    /// <summary>
    /// output.
    /// </summary>
    Output = 1,

    /// <summary>
    /// constant data.
    /// </summary>
    Rdata = 2,

    /// <summary>
    /// compute temp data.
    /// </summary>
    Data = 3,

    /// <summary>
    /// shared data.
    /// </summary>
    SharedData = 4,

    /// <summary>
    /// base addr.
    /// </summary>
    PrivateBase = 64,
}

/// <summary>
/// memory range define.
/// </summary>
public class MemoryRange
{
    /// <summary>
    /// memory loaction.
    /// </summary>
    public MemoryLocation MemoryLocate;

    /// <summary>
    /// memory data type.
    /// </summary>
    public DataType DType;

    /// <summary>
    /// shared module.
    /// </summary>
    public ushort SharedModule;

    /// <summary>
    /// memory span start.
    /// </summary>
    public uint Start;

    /// <summary>
    /// memory span length.
    /// </summary>
    public uint Size;

    /// <summary>
    /// <see cref="MemoryRange"/>.
    /// </summary>
    /// <param name="memoryLocate">memory data type.</param>
    /// <param name="dType">memory loaction.</param>
    /// <param name="sharedModule">shared module.</param>
    /// <param name="start">memory span start.</param>
    /// <param name="size">memory span length.</param>
    public MemoryRange(MemoryLocation memoryLocate, DataType dType, ushort sharedModule, uint start, uint size)
    {
        MemoryLocate = memoryLocate;
        DType = dType;
        SharedModule = sharedModule;
        Start = start;
        Size = size;
    }
}

/// <summary>
/// the buffer allocation.
/// </summary>
public class BufferAllocation
{
    /// <summary>
    /// mem loacte.
    /// </summary>
    public MemoryLocation MemoryLocate;

    /// <summary>
    /// data type.
    /// </summary>
    public DataType DType;

    /// <summary>
    /// shared modeule.
    /// </summary>
    public ulong SharedModule;

    /// <summary>
    /// start.
    /// </summary>
    public ulong Start;

    /// <summary>
    /// total size.
    /// </summary>
    public ulong Size;

    /// <summary>
    /// full shape.
    /// </summary>
    public IR.Shape Shape;

    /// <summary>
    /// full stride.
    /// </summary>
    public IR.Shape Strides;

    /// <summary>
    /// stride shape.
    /// </summary>
    public IR.Shape StridesShape;

    /// <summary>
    /// <see cref="BufferAllocation"/>.
    /// </summary>
    /// <param name="memory_locate">mem loacte.</param>
    /// <param name="d_type">data type.</param>
    /// <param name="shared_module">shared modeule.</param>
    /// <param name="start">start.</param>
    /// <param name="size">total size.</param>
    /// <param name="shape">full shape.</param>
    /// <param name="strides">full stride.</param>
    /// <param name="strides_shape">stride shape.</param>
    public BufferAllocation(MemoryLocation memory_locate, DataType d_type, ulong shared_module, ulong start, ulong size, IR.Shape shape, IR.Shape strides, IR.Shape strides_shape)
    {
        MemoryLocate = memory_locate;
        DType = d_type;
        SharedModule = shared_module;
        Start = start;
        Size = size;
        Shape = shape;
        Strides = strides;
        StridesShape = strides_shape;
    }

    /// <summary>
    /// get then mem span end.
    /// </summary>
    /// <returns></returns>
    public ulong LinearEnd() => Start + Size;

    /// <summary>
    /// calc the overlap with another buffer.
    /// </summary>
    /// <param name="rhs"></param>
    /// <returns></returns>
    public bool Overlap(BufferAllocation rhs) => Size != 0 && rhs.Size != 0 && this.MemoryLocate == rhs.MemoryLocate && (Start < rhs.LinearEnd() && LinearEnd() > rhs.Start);

    /// <summary>
    /// get current buffer memory range.
    /// </summary>
    public MemoryRange RuntimeType => new(this.MemoryLocate,
         this.DType,
         (ushort)SharedModule,
         (uint)this.Start,
         (uint)this.Size);
}

/// <summary>
/// SchedModelResult.
/// </summary>
public class SchedModelResult
{
    /// <summary>
    /// sched module result.
    /// </summary>
    public readonly List<SchedModuleResult> Modules;

    /// <summary>
    /// sched function result.
    /// </summary>
    public SchedFunctionResult? Entry;

    /// <summary>
    /// the parent ir module.
    /// </summary>
    public IR.IRModule ParentModule;

    /// <summary>
    /// create the SchedModelResult.
    /// </summary>
    /// <param name="parent_module"></param>
    public SchedModelResult(IR.IRModule parent_module)
    {
        ParentModule = parent_module;
        Modules = new();
        Entry = null;
    }
}

/// <summary>
/// SchedModuleResult.
/// </summary>
public class SchedModuleResult
{
    /// <summary>
    /// current Module type.
    /// </summary>
    public CodeGen.ModuleType ModuleType;

    /// <summary>
    /// contains functions.
    /// </summary>
    public readonly List<SchedFunctionResult> Functions;

    /// <summary>
    /// schedfunction maps.
    /// </summary>
    public readonly Dictionary<IR.Expr, SchedFunctionResult> FunctionsMap;

    /// <summary>
    /// the buffer allocations.
    /// </summary>
    public readonly AllocationMap Allocations;

    /// <summary>
    /// mem collection.
    /// </summary>
    public readonly Dictionary<MemoryLocation, ulong> MaxUsages;

    /// <summary>
    /// shared mem.
    /// </summary>
    public readonly Dictionary<CodeGen.ModuleType, ulong> SharedMaxUsages;

    /// <summary>
    /// create SchedModuleResult.
    /// </summary>
    public SchedModuleResult(CodeGen.ModuleType moduleType)
    {
        ModuleType = moduleType;
        Functions = new();
        FunctionsMap = new();
        Allocations = new();
        MaxUsages = new();
        SharedMaxUsages = new();
    }
}

/// <summary>
/// SchedFunctionResult.
/// </summary>
public class SchedFunctionResult
{
    /// <summary>
    /// parent module.
    /// </summary>
    public SchedModuleResult SchedModule;

    /// <summary>
    /// input memory size.
    /// </summary>
    public ulong InputPoolSize;

    /// <summary>
    /// ouput memory size.
    /// </summary>
    public ulong OutputPoolSize;

    /// <summary>
    /// compute sequence.
    /// </summary>
    public List<IR.Expr> ComputeSequence;

    /// <summary>
    /// the ir function.
    /// </summary>
    public IR.Function Function;

    /// <summary>
    /// create SchedFunctionResult.
    /// </summary>
    /// <param name="sched_module">parent module.</param>
    /// <param name="input_pool_size">input memory size.</param>
    /// <param name="output_pool_size">ouput memory size.</param>
    /// <param name="function">the ir function.</param>
    public SchedFunctionResult(SchedModuleResult sched_module, ulong input_pool_size, ulong output_pool_size, IR.Function function)
    {
        SchedModule = sched_module;
        InputPoolSize = input_pool_size;
        OutputPoolSize = output_pool_size;
        ComputeSequence = new();
        Function = function;
    }
}

/// <summary>
/// the scheduler interface.
/// </summary>
public interface IScheduler
{
    /// <summary>
    /// the current target.
    /// </summary>
    public ITarget Target { get; set; }

    /// <summary>
    /// the main module.
    /// </summary>
    public IR.IRModule ParentModule { get; set; }

    /// <summary>
    /// multi stage schedules.
    /// relay IR -> TIR -> lowered TIR.
    /// </summary>
    /// <param name="skip_buffer_alias"></param>
    /// <returns></returns>
    public SchedModelResult Schedule(bool skip_buffer_alias = false);
}
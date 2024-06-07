// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections;
using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Runtime.InteropServices;
using Nncase.Runtime;
using Nncase.TIR;

namespace Nncase.Schedule;

/// <summary>
/// the scheduler interface.
/// </summary>
public interface IScheduler
{
    /// <summary>
    /// Gets or sets the current target.
    /// </summary>
    public ITarget Target { get; set; }

    /// <summary>
    /// Gets or sets the main module.
    /// </summary>
    public IR.IRModule Module { get; set; }

    /// <summary>
    /// multi stage schedules.
    /// relay IR -> TIR -> lowered TIR.
    /// </summary>
    public IR.IRModule Schedule(bool skip_buffer_alias = false);
}

/// <summary>
/// memory range define.
/// </summary>
[StructLayout(LayoutKind.Sequential)]
public struct MemoryRange
{
    /// <summary>
    /// memory loaction.
    /// </summary>
    public MemoryLocation MemoryLocate;

    /// <summary>
    /// memory data type.
    /// </summary>
    public PrimTypeCode DType;

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
    /// Initializes a new instance of the <see cref="MemoryRange"/> struct.
    /// <see cref="MemoryRange"/>.
    /// </summary>
    /// <param name="memoryLocate">memory data type.</param>
    /// <param name="dType">memory loaction.</param>
    /// <param name="sharedModule">shared module.</param>
    /// <param name="start">memory span start.</param>
    /// <param name="size">memory span length.</param>
    public MemoryRange(MemoryLocation memoryLocate, PrimTypeCode dType, ushort sharedModule, uint start, uint size)
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
    /// Initializes a new instance of the <see cref="BufferAllocation"/> class.
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
    public BufferAllocation(MemoryLocation memory_locate, DataType d_type, ulong shared_module, ulong start, ulong size, int[] shape, int[] strides, int[] strides_shape)
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
    /// Gets or sets mem loacte.
    /// </summary>
    public MemoryLocation MemoryLocate { get; set; }

    /// <summary>
    /// Gets or sets data type.
    /// </summary>
    public DataType DType { get; set; }

    /// <summary>
    /// Gets or sets shared modeule.
    /// </summary>
    public ulong SharedModule { get; set; }

    /// <summary>
    /// Gets or sets start.
    /// </summary>
    public ulong Start { get; set; }

    /// <summary>
    /// Gets or sets total size.
    /// </summary>
    public ulong Size { get; set; }

    /// <summary>
    /// Gets or sets full shape.
    /// </summary>
    public int[] Shape { get; set; }

    /// <summary>
    /// Gets or sets full stride.
    /// </summary>
    public int[] Strides { get; set; }

    /// <summary>
    /// Gets or sets stride shape.
    /// </summary>
    public int[] StridesShape { get; set; }

    /// <summary>
    /// Gets get current buffer memory range.
    /// </summary>
    public MemoryRange MemoryRange
    {
        get
        {
            // todo because of
            PrimTypeCode code;
            try
            {
                code = PrimTypeCodes.ToTypeCode(DType);
            }
            catch (System.Collections.Generic.KeyNotFoundException)
            {
                if (DType.SizeInBytes == 4)
                {
                    code = PrimTypeCode.Float32;
                }
                else
                {
                    throw;
                }
            }

            return new(
                MemoryLocate,
                code,
                (ushort)SharedModule,
                (uint)Start,
                (uint)Size);
        }
    }

    /// <summary>
    /// get then mem span end.
    /// </summary>
    public ulong LinearEnd() => Start + Size;

    /// <summary>
    /// calc the overlap with another buffer.
    /// </summary>
    public bool Overlap(BufferAllocation rhs) => Size != 0 && rhs.Size != 0 && MemoryLocate == rhs.MemoryLocate && Start < rhs.LinearEnd() && LinearEnd() > rhs.Start;
}

/// <summary>
/// SchedFunctionResult.
/// </summary>
public sealed class SchedFunctionResult
{
    /// <summary>
    /// Initializes a new instance of the <see cref="SchedFunctionResult"/> class.
    /// create SchedFunctionResult.
    /// </summary>
    public SchedFunctionResult()
    {
        Rdatas = new(ReferenceEqualityComparer.Instance);
        DataUsage = 0;
        IsScheduled = false;
    }

    /// <summary>
    /// Gets the buffer allocation.
    /// </summary>
    public Dictionary<IR.Const, ValueRange<long>> Rdatas { get; }

    /// <summary>
    /// Gets or sets the data section length.
    /// </summary>
    public ulong DataUsage { get; set; }

    /// <summary>
    /// Gets or sets the data section align.
    /// </summary>
    public ulong DataAlign { get; set; }

    /// <summary>
    /// Gets or sets a value indicating whether the Scheduled status.
    /// </summary>
    public bool IsScheduled { get; set; }

    /// <inheritdoc/>
    public override bool Equals(object? obj)
    {
        if (obj is not SchedFunctionResult result)
        {
            return false;
        }

        if (IsScheduled != result.IsScheduled)
        {
            return false;
        }

        if (Rdatas.Count != result.Rdatas.Count)
        {
            return false;
        }

        if (Rdatas.Count == 0)
        {
            return true;
        }

        return EqualityComparer<Dictionary<IR.Const, ValueRange<long>>>.Default.Equals(Rdatas, result.Rdatas) &&
               EqualityComparer<ulong>.Default.Equals(DataUsage, result.DataUsage) &&
               EqualityComparer<ulong>.Default.Equals(DataAlign, result.DataAlign);
    }

    /// <inheritdoc/>
    public override int GetHashCode() => base.GetHashCode();
}

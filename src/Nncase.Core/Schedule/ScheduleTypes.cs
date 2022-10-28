// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Collections.Generic;
using System.ComponentModel.DataAnnotations;
using System.Runtime.InteropServices;
using Nncase.Runtime;

namespace Nncase.Schedule;


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
    /// l2 data
    /// </summary>
    L2Data = 5,

    /// <summary>
    /// L1 data
    /// </summary>
    L1Data = 6,

    /// <summary>
    /// base addr.
    /// </summary>
    PrivateBase = 64,
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
    /// <see cref="MemoryRange"/>.
    /// </summary>
    /// <param name="memoryLocate">memory data type.</param>
    /// <param name="dType">memory loaction.</param>
    /// <param name="sharedModule">shared module.</param>
    /// <param name="start">memory span start.</param>
    /// <param name="size">memory span length.</param>
    public MemoryRange(MemoryLocation memoryLocate, PrimTypeCode dType, UInt16 sharedModule, uint start, uint size)
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
    public int[] Shape;

    /// <summary>
    /// full stride.
    /// </summary>
    public int[] Strides;

    /// <summary>
    /// stride shape.
    /// </summary>
    public int[] StridesShape;

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
            catch (System.Collections.Generic.KeyNotFoundException e)
            {
                if (DType.SizeInBytes == 4)
                    code = PrimTypeCode.Float32;
                else
                    throw e;
            }
            return new(this.MemoryLocate,
               code,
               (UInt16)SharedModule,
               (uint)this.Start,
               (uint)this.Size);
        }
    }
}


/// <summary>
/// SchedFunctionResult.
/// </summary>
public class SchedFunctionResult
{
    /// <summary>
    /// the buffer allocation
    /// </summary>
    public readonly HashSet<TIR.PhysicalBuffer> Rdatas;

    /// <summary>
    /// the Scheduled status
    /// </summary>
    public bool IsScheduled;

    /// <summary>
    /// create SchedFunctionResult
    /// </summary>
    public SchedFunctionResult()
    {
        Rdatas = new(ReferenceEqualityComparer.Instance);
        IsScheduled = false;
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
    public IR.IRModule Module { get; set; }

    /// <summary>
    /// multi stage schedules.
    /// relay IR -> TIR -> lowered TIR.
    /// </summary>
    /// <param name="skip_buffer_alias"></param>
    /// <returns></returns>
    public IR.IRModule Schedule(bool skip_buffer_alias = false);
}
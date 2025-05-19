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

public record struct Interval
{
    public Interval(long start, long end)
    {
        Start = start;
        Stop = end;
    }

    public long Start { get; set; }

    public long Stop { get; set; }

    public long Size
    {
        get => Stop - Start;
        set => Stop = Start + value;
    }

    public bool Overlaps(Interval rhs) => Size != 0 && rhs.Size != 0 && Start < Stop && Stop > rhs.Start;

    public override string ToString()
    {
        return $"Interval({Start}, {Stop})";
    }
}

public record class BufferLifetime
{
    public Interval Time;

    public Interval Memory;

    public BufferLifetime(TIR.Buffer buffer)
    {
        Buffer = buffer;
    }

    public TIR.Buffer Buffer { get; }
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
        LocalRdatas = new(ReferenceEqualityComparer.Instance);
        DataUsage = 0;
        IsScheduled = false;
    }

    /// <summary>
    /// Gets the buffer allocation.
    /// </summary>
    public Dictionary<IR.Const, ValueRange<ulong>> Rdatas { get; }

    /// <summary>
    /// Gets the buffer allocation.
    /// </summary>
    public Dictionary<IR.Const, ValueRange<ulong>> LocalRdatas { get; }

    /// <summary>
    /// Gets or sets the data section length.
    /// </summary>
    public ulong DataUsage { get; set; }

    /// <summary>
    /// Gets or sets the data section align.
    /// </summary>
    public ulong DataAlign { get; set; } = 8;

    public ulong OutputUsage { get; set; }

    public ulong OutputAlign { get; set; } = 8;

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

        return EqualityComparer<Dictionary<IR.Const, ValueRange<ulong>>>.Default.Equals(Rdatas, result.Rdatas) &&
               EqualityComparer<ulong>.Default.Equals(DataUsage, result.DataUsage) &&
               EqualityComparer<ulong>.Default.Equals(DataAlign, result.DataAlign);
    }

    /// <inheritdoc/>
    public override int GetHashCode() => base.GetHashCode();
}

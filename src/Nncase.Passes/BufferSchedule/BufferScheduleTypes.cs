// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.BufferSchedule;

internal sealed class TimeInterval
{
    public TimeInterval(int start, int end)
    {
        Start = start;
        End = end;
    }

    public int Start { get; set; }

    public int End { get; set; }

    public override string ToString()
    {
        return $"TimeInterval({Start}, {End})";
    }
}

internal sealed class MemSpan
{
    public MemSpan(int start, int end)
    {
        Start = start;
        End = end;
    }

    public int Start { get; set; }

    public int End { get; set; }

    public override string ToString()
    {
        return $"MemSpan({Start}, {End})";
    }
}

internal class ScheduleBuffer
{
    public ScheduleBuffer(string name, TimeInterval interval, MemSpan span, int[] shape, int[] strides)
    {
        Name = name;
        Interval = interval;
        Span = span;
        Shape = shape;
        Strides = strides;
    }

    public string Name { get; }

    public TimeInterval Interval { get; }

    public MemSpan Span { get; }

    public int[] Shape { get; }

    public int[] Strides { get; }

    public override string ToString()
    {
        return $"ScheduledBuffer('{Name}', {Interval}, {Span}, ConstraintsMode.No, [{string.Join(",", Shape)}], [{string.Join(",", Strides)}])";
    }
}

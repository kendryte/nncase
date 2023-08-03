// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

namespace Nncase.Passes.BufferSchedule;

internal sealed class TimeInterval
{
    public TimeInterval(int start, int end)
    {
        Brith = start;
        Death = end;
    }

    public int Brith { get; set; }

    public int Death { get; set; }

    public int Size => Death - Brith;

    public override string ToString()
    {
        return $"TimeInterval({Brith}, {Death})";
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

    public int Size => End - Start;

    public override string ToString()
    {
        return $"MemSpan({Start}, {End})";
    }
}

internal class ScheduleBuffer
{
    public ScheduleBuffer(string name, int number, TimeInterval interval, MemSpan span, int[] shape, int[] strides, bool inplace)
    {
        Name = name;
        Number = number;
        Interval = interval;
        Span = span;
        Shape = shape;
        Strides = strides;
        Inplace = inplace;
    }

    public string Name { get; }
    public int Number { get; }
    public TimeInterval Interval { get; }

    public MemSpan Span { get; }

    public int[] Shape { get; }

    public int[] Strides { get; }

    public bool Inplace { get; }

    public override string ToString()
    {
        return $"ScheduledBuffer('{Name}', {Number}, {Interval}, {Span}, ConstraintsMode.No, [{string.Join(",", Shape)}], [{string.Join(",", Strides)}], {Inplace})";
    }
}

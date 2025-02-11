﻿// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;

namespace Nncase.Passes.BufferSchedule;

public sealed class Interval
{
    public Interval(long start, long end)
    {
        Start = start;
        Stop = end;
    }

    public long Start { get; set; }

    public long Stop { get; set; }

    public long Size => Stop - Start;

    public override string ToString()
    {
        return $"Interval({Start}, {Stop})";
    }
}

public class ScheduleBuffer
{
    public ScheduleBuffer(string name, int number, Interval timeInterval, Interval memInterval, Shape shape, long[] strides, bool inplace)
    {
        Name = name;
        Number = number;
        TimeInterval = timeInterval;
        MemInterval = memInterval;
        Shape = shape;
        Strides = strides;
        Inplace = inplace;
    }

    public string Name { get; }

    public int Number { get; }

    public Interval TimeInterval { get; }

    public Interval MemInterval { get; }

    public Shape Shape { get; }

    public long[] Strides { get; }

    public bool Inplace { get; }

    public override string ToString()
    {
        return $"ScheduledBuffer('{Name}', {Number}, {TimeInterval}, {MemInterval}, ConstraintsMode.No, [{string.Join(",", Shape)}], [{string.Join(",", Strides)}], {Inplace})";
    }
}

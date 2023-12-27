// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text.RegularExpressions;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.TIR;
using Buffer = Nncase.TIR.Buffer;

namespace Nncase.Passes.BufferSchedule;

internal class Lifeness
{
    public Lifeness(int start, int end)
    {
        Start = start;
        End = end;
    }

    public int Start { get; set; }

    public int End { get; set; }

    public override string ToString()
    {
        return $"Lifeness({Start}, {End})";
    }
}

internal class ScheduledBuffer
{
    public ScheduledBuffer(Lifeness lifeness, Buffer buffer)
    {
        Lifeness = lifeness;
        Buffer = buffer;
    }

    public Lifeness Lifeness { get; }

    public Buffer Buffer { get; }

    public string Name => Buffer.Name;

    public override string ToString()
    {
        return $"ScheduledBuffer(\"{Name}\", {Lifeness}, Location({Buffer.MemSpan.Start}, {Buffer.MemSpan.Size}), [{string.Join(",", Buffer.Dimensions.ToArray().Select(s => ((TensorConst)s).Value[0]))}], [{string.Join(",", Buffer.Strides.ToArray().Select(s => ((TensorConst)s).Value[0]))}])";
    }
}

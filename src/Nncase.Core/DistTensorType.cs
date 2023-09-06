// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.
using System;
using System.Collections;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;

namespace Nncase.IR;

public abstract record SBP
{
    public static SBPSumParallel P => SBPSumParallel.Instance;

    public static SBPSplit S(int axis) => new SBPSplit(axis);

    public static SBPBroadCast B => SBPBroadCast.Instance;
}

public sealed record SBPSplit(int Axis) : SBP
{
    public override string ToString() => $"S({Axis})";
}

public sealed record SBPSumParallel : SBP
{
    public static readonly SBPSumParallel Instance = new SBPSumParallel();

    private SBPSumParallel()
    {
    }

    public override string ToString() => "P";
}

public sealed record SBPBroadCast : SBP
{
    public static readonly SBPBroadCast Instance = new SBPBroadCast();

    private SBPBroadCast()
    {
    }

    public override string ToString() => "B";
}

public sealed record Placement
{
    public Placement(DeviceKind kind, IRArray<int> hierarchy, string name)
    {
        Kind = kind;
        Hierarchy = hierarchy;
        Name = name;
    }

    /// <summary>
    /// The device kind type.
    /// </summary>
    public enum DeviceKind : uint
    {
        CPU = 0,
    }

    public DeviceKind Kind { get; }

    public IRArray<int> Hierarchy { get; }

    public string Name { get; }

    public int Rank => Hierarchy.Count;

    public override string ToString() => $"@{Kind} [{string.Join(',', Hierarchy.Zip(Name).Select(t => t.First.ToString() + '@' + t.Second.ToString()))}]";
}

public sealed record DistTensorType : IRType
{
    public DistTensorType(TensorType tensorType, IRArray<SBP> ndsbp, Placement placement)
    {
        if (placement.Hierarchy.Count != ndsbp.Count)
        {
            throw new ArgumentException("spb dimension != placement rank!");
        }

        TensorType = tensorType;
        NdSbp = ndsbp;
        Placement = placement;
    }

    public TensorType TensorType { get; }

    public IRArray<SBP> NdSbp { get; }

    public Placement Placement { get; }

    public override string ToString() => $"{TensorType} ({string.Join(',', NdSbp)}) {Placement}";
}

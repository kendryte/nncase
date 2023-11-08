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
    public static SBPPartialSum P => SBPPartialSum.Instance;

    public static SBPBroadCast B => SBPBroadCast.Instance;

    public static SBPSplit S(int axis) => new SBPSplit(axis);
}

public sealed record SBPSplit(int Axis) : SBP
{
    public override string ToString() => $"S({Axis})";
}

public sealed record SBPPartialSum : SBP
{
    public static readonly SBPPartialSum Instance = new SBPPartialSum();

    private SBPPartialSum()
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

// public sealed record Placement(Placement.DeviceKind Kind, IRArray<int> Hierarchy, string Name)
public sealed record Placement(IRArray<int> Hierarchy, string Name)
{
    // public enum DeviceKind : uint
    // {
    //     CPU = 0,
    // }
    public int Rank => Hierarchy.Count;

    // public override string ToString() => $"@{Kind} [{string.Join(',', Hierarchy.Zip(Name).Select(t => t.First.ToString() + '@' + t.Second.ToString()))}]";
    public override string ToString() => $"@ [{string.Join(',', Hierarchy.Zip(Name).Select(t => t.First.ToString() + '@' + t.Second.ToString()))}]";
}

public sealed record DistributedType(TensorType TensorType, IRArray<SBP> NdSBP, Placement Placement) : IRType
{
    public override string ToString() => $"{TensorType}, ({string.Join(',', NdSBP)}), {Placement}";
}

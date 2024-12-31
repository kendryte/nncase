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
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using DryIoc.ImTools;

namespace Nncase.IR;

public enum HierarchyKind : byte
{
    Parallel = 0,
    SMT = 1,
}

[JsonDerivedType(typeof(SBPSplit), "S")]
[JsonDerivedType(typeof(SBPPartial), "P")]
[JsonDerivedType(typeof(SBPBroadCast), "B")]
public abstract record SBP
{
    public static SBPBroadCast B => SBPBroadCast.Instance;

    public static SBPPartial P(ReduceOp op = ReduceOp.Sum) => new SBPPartial(op);

    public static SBPSplit S(int axis) => new SBPSplit(axis);
}

public sealed record SBPSplit(int Axis) : SBP
{
    public override string ToString() => $"S({Axis})";
}

public sealed record SBPPartial(ReduceOp Op) : SBP
{
    public override string ToString() => $"P({Op})";
}

public sealed record SBPBroadCast : SBP
{
    public static readonly SBPBroadCast Instance = new SBPBroadCast();

    public override string ToString() => "B";
}

// public sealed record Placement(Placement.DeviceKind Kind, IRArray<int> Hierarchy, string Name, HierarchyKind HierarchyKind)
public sealed record Placement(IRArray<int> Hierarchy, string Name, HierarchyKind HierarchyKind = HierarchyKind.Parallel)
{
    // public enum DeviceKind : uint
    // {
    //     CPU = 0,
    // }
    public int Rank => Hierarchy.Count;

    public override string ToString() => $"[{string.Join(',', Hierarchy.Zip(Name).Select(t => t.Second.ToString() + ':' + t.First.ToString()))}]";
}

public sealed record DistributedType(TensorType TensorType, IRArray<SBP> NdSBP, Placement Placement) : IRType
{
    public override string ToString() => $"{TensorType}, ({string.Join(',', NdSBP)}), {Placement}";
}

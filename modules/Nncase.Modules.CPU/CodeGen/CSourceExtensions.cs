// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using Nncase.IR;
using Nncase.TIR;

namespace Nncase.CodeGen.CPU;

/// <summary>
/// convert the type/op to c name.
/// </summary>
internal static class CSourceExtensions
{
    private static readonly Dictionary<PrimType, string> _primTypeToC = new()
    {
        { DataTypes.Boolean, "uint8_t" },
        { DataTypes.Int8, "int8_t" },
        { DataTypes.Int16, "int16_t" },
        { DataTypes.Int32, "int32_t" },
        { DataTypes.Int64, "int64_t" },
        { DataTypes.UInt8, "uint8_t" },
        { DataTypes.UInt16, "uint16_t" },
        { DataTypes.UInt32, "uint32_t" },
        { DataTypes.UInt64, "uint64_t" },
        { DataTypes.Float32, "float" },
        { DataTypes.Float64, "double" },
    };

    public static string ToC(this PrimType primType) =>
        _primTypeToC[primType];

    public static string ToC(this DataType dataType) => dataType switch
    {
        PrimType ptype => ptype.ToC(),
        PointerType => "uint8_t *",
        _ => throw new NotSupportedException(dataType.ToString()),
    };

    public static string ToC(this MemoryLocation location) => location switch
    {
        MemoryLocation.Output or MemoryLocation.Input or MemoryLocation.Rdata => "loc_t::device",
        MemoryLocation.L2Data => "loc_t::shared",
        MemoryLocation.L1Data => "loc_t::local",
        _ => throw new NotSupportedException(location.ToString()),
    };

    public static string ToSlicing(this TensorType tensorType, string[] begins, IRArray<SBP> ndsbp, Placement placement)
    {
        var hstrides = TensorUtilities.GetStrides(placement.Hierarchy.ToArray());
        var splits = Enumerable.Range(0, begins.Length).Select(_ => new List<(int H, SBPSplit S)>()).ToArray();
        foreach (var (sbp, i) in ndsbp.Select((s, i) => (s, i)))
        {
            if (sbp is SBPSplit { Axis: int axis } split)
            {
                splits[axis].Add((i, split));
            }
        }

        foreach (var splist in splits)
        {
            splist.Sort((a, b) => -a.H.CompareTo(b.H));
        }

        for (int i = 0; i < begins.Length; i++)
        {
            var sp = splits[i];
            if (sp.Count > 0)
            {
                begins[i] += " + " + sp.Skip(1).Aggregate($"{placement.Name[sp[0].H]}id", (acc, p) => $"({acc} + {TensorUtilities.GetProduct(placement.Hierarchy[(p.H + 1)..])} * {placement.Name[p.H]}id)");
            }
        }

        return $"({{{string.Join(',', begins)}}}, {{{string.Join(",", tensorType.Shape)}}})";
    }

    public static string ToSlicing(this TensorType tensorType, IRArray<SBP> ndsbp, Placement placement) => ToSlicing(tensorType, Enumerable.Repeat("0", tensorType.Shape.Rank).ToArray(), ndsbp, placement);

    public static string ToC(this BinaryOp binaryOp) => binaryOp switch
    {
        BinaryOp.Add => "+",
        BinaryOp.Sub => "-",
        BinaryOp.Mul => "*",
        BinaryOp.Div => "/",
        _ => throw new NotSupportedException(binaryOp.ToString()),
    };
}

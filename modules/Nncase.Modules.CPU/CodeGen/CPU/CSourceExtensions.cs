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

    public static string ToC(this ReduceArgOp op) => op switch
    {
        ReduceArgOp.ArgMin => "arg_min",
        ReduceArgOp.ArgMax => "arg_max",
        _ => throw new NotImplementedException(),
    };

    public static string ToC(this DataType dataType) => dataType switch
    {
        PrimType ptype => ptype.ToC(),
        PointerType => "uint8_t *",
        VectorType vtype => $"vector<{vtype.ElemType.ToC()},{string.Join(",", vtype.Lanes)}>",
        _ => throw new NotSupportedException(dataType.ToString()),
    };

    public static string ToC(this MemoryLocation location) => location switch
    {
        MemoryLocation.Output or MemoryLocation.Input or MemoryLocation.Rdata => "loc_t::device",
        MemoryLocation.L2Data => "loc_t::shared",
        MemoryLocation.L1Data => "loc_t::local",
        _ => throw new NotSupportedException(location.ToString()),
    };

    public static string ToC(this ImageResizeMode mode) => mode switch
    {
        ImageResizeMode.Bilinear => "bilinear",
        ImageResizeMode.NearestNeighbor => "nearest_neighbor",
        _ => throw new NotImplementedException(),
    };

    public static string ToC(this ImageResizeTransformationMode mode) => mode switch
    {
        ImageResizeTransformationMode.HalfPixel => "half_pixel",
        ImageResizeTransformationMode.PytorchHalfPixel => "pytorch_half_pixel",
        ImageResizeTransformationMode.AlignCorners => "align_corners",
        ImageResizeTransformationMode.Asymmetric => "asymmetric",
        ImageResizeTransformationMode.TFCropAndResize => "tfcrop_and_resize",
        _ => throw new NotImplementedException(),
    };

    public static string ToC(this ImageResizeNearestMode mode) => mode switch
    {
        ImageResizeNearestMode.RoundPreferFloor => "round_prefer_floor",
        ImageResizeNearestMode.RoundPreferCeil => "round_prefer_ceil",
        ImageResizeNearestMode.Floor => "floor",
        ImageResizeNearestMode.Ceil => "ceil",
        _ => throw new NotImplementedException(),
    };

    public static string ToSlicing(this IEnumerable<string> dims, string[] begins, IRArray<SBP> ndsbp, Placement placement)
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
                var dimi = dims.ElementAt(i);
                if (dimi.IndexOf('?', System.StringComparison.CurrentCulture) is int s && dimi.IndexOf(':', System.StringComparison.CurrentCulture) is int e && s != -1 && e != -1)
                {
                    dimi = dimi[(s + 1)..e].Trim();
                }

                begins[i] += " + " + sp.Skip(1).Aggregate($"{placement.Name[sp[0].H]}id", (acc, p) => $"({acc} + {TensorUtilities.GetProduct(placement.Hierarchy[(p.H + 1)..])} * {placement.Name[p.H]}id)") + $" * {dimi}";
            }
        }

        return $".view(make_ranked_shape({string.Join(',', begins)}), fixed_shape<{string.Join(",", dims.Select(d => d.ToString()))}>{{}})";
    }

    public static string ToSlicing(this IEnumerable<string> dims, IRArray<SBP> ndsbp, Placement placement) => ToSlicing(dims, Enumerable.Repeat("0", dims.Count()).ToArray(), ndsbp, placement);

    public static string ToC(this BinaryOp binaryOp) => binaryOp switch
    {
        BinaryOp.Add => "+",
        BinaryOp.Sub => "-",
        BinaryOp.Mul => "*",
        BinaryOp.Div => "/",
        _ => throw new NotSupportedException(binaryOp.ToString()),
    };

    public static string ToC(this CompareOp op) => op switch
    {
        CompareOp.Equal => "==",
        CompareOp.NotEqual => "!=",
        CompareOp.LowerThan => "<",
        CompareOp.LowerOrEqual => "<=",
        CompareOp.GreaterThan => ">=",
        CompareOp.GreaterOrEqual => ">",
        _ => throw new NotSupportedException(op.ToString()),
    };
}

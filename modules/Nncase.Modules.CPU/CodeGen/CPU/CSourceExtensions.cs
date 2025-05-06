// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using DryIoc.ImTools;
using Nncase.IR;
using Nncase.IR.NN;
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
        { DataTypes.Float16, "nncase::half" },
        { DataTypes.BFloat16, "nncase::bfloat16" },
        { DataTypes.Float32, "float" },
        { DataTypes.Float64, "double" },
        { DataTypes.Float8E4M3, "nncase::float_e4m3_t" },
        { DataTypes.Float8E5M2, "nncase::float_e5m2_t" },
    };

    public static string ToC(this PrimType primType) =>
        _primTypeToC[primType];

    public static string ToC(this ReduceArgOp op) => op switch
    {
        ReduceArgOp.ArgMin => "arg_min",
        ReduceArgOp.ArgMax => "arg_max",
        _ => throw new NotImplementedException(),
    };

    public static string ToC(this ReduceOp op) => op switch
    {
        ReduceOp.Min => "min",
        ReduceOp.Max => "max",
        ReduceOp.Sum => "sum",
        ReduceOp.Mean => "mean",
        ReduceOp.Prod => "prod",
        _ => throw new NotImplementedException(),
    };

    public static string ToC(this IR.NN.AttentionCacheKind mode) => mode switch
    {
        IR.NN.AttentionCacheKind.Key => "caching::attention_cache_kind::key",
        IR.NN.AttentionCacheKind.Value => "caching::attention_cache_kind::value",
        _ => throw new NotImplementedException(),
    };

    public static string ToC(this IR.NN.PagedAttentionDimKind mode) => mode switch
    {
        IR.NN.PagedAttentionDimKind.NumBlocks => "caching::paged_attention_dim_kind::num_blocks",
        IR.NN.PagedAttentionDimKind.NumLayers => "caching::paged_attention_dim_kind::num_layers",
        IR.NN.PagedAttentionDimKind.KV => "caching::paged_attention_dim_kind::kv",
        IR.NN.PagedAttentionDimKind.BlockSize => "caching::paged_attention_dim_kind::block_size",
        IR.NN.PagedAttentionDimKind.NumKVHeads => "caching::paged_attention_dim_kind::num_kv_heads",
        IR.NN.PagedAttentionDimKind.HeadDim => "caching::paged_attention_dim_kind::head_dim",
        _ => throw new NotImplementedException(),
    };

    public static string ToC(this DataType dataType) => dataType switch
    {
        PrimType ptype => ptype.ToC(),
        IR.NN.PagedAttentionKVCacheType kv_type => $"caching::paged_attention_kv_cache<caching::paged_attention_config<{kv_type.Config.NumLayers}, {kv_type.Config.NumKVHeads}, {kv_type.Config.HeadDim}, {kv_type.Config.KVType.ToC()}, {kv_type.Config.BlockSize}, fixed_shape<{string.Join(',', kv_type.Config.CacheLayout.Select(e => "(size_t)" + e.ToC()))}>, fixed_shape<{string.Join(',', kv_type.Config.BlockLayout.Select(e => "(size_t)" + e.ToC()))}>, fixed_shape<{string.Join(',', kv_type.Config.PackedAxes.Select(e => "(size_t)" + e.ToC()))}>, fixed_shape<{string.Join(',', kv_type.Config.Lanes)}>, fixed_shape<{string.Join(',', kv_type.Config.Topology)}>>>",
        PointerType => "uint8_t *",
        VectorType vtype => $"vector<{vtype.ElemType.ToC()},{string.Join(",", vtype.Lanes)}>",
        ReferenceType rtype => $"{rtype.ElemType.ToC()}",
        _ => throw new NotSupportedException(dataType.ToString()),
    };

    public static string ToC(this PagedAttentionKVCacheType pagedAttentionKVCacheType)
    {
        var config = pagedAttentionKVCacheType.Config;
        var configStr = $"ntt::caching::paged_attention_config<{config.NumLayers}, {config.NumKVHeads}, {config.HeadDim}, {config.KVType.ToC()}, {config.BlockSize}, {config.CacheLayout.ToC()}, {config.BlockLayout.ToC()}, {config.PackedAxes.ToC()}, fixed_shape<{string.Join(", ", config.Lanes)}>, fixed_shape<{string.Join(", ", config.Topology)}>>";
        return $"ntt::caching::paged_attention_kv_cache<{configStr}>";
    }

    public static string ToC(this IEnumerable<PagedAttentionDimKind> layout)
    {
        var layoutStr = string.Join(", ", layout.Select(x => $"ntt::caching::paged_attention_dim_kind::{x.ToC()}"));
        return $"fixed_dims<ntt::caching::paged_attention_dim_kind, {layoutStr}>";
    }

    public static string ToC(this PagedAttentionDimKind kind) => kind switch
    {
        PagedAttentionDimKind.NumBlocks => "num_blocks",
        PagedAttentionDimKind.NumLayers => "num_layers",
        PagedAttentionDimKind.KV => "kv",
        PagedAttentionDimKind.BlockSize => "block_size",
        PagedAttentionDimKind.NumKVHeads => "num_kv_heads",
        PagedAttentionDimKind.HeadDim => "head_dim",
        _ => throw new NotSupportedException(kind.ToString()),
    };

    public static string ToC(this ImageResizeMode mode) => mode switch
    {
        ImageResizeMode.Bilinear => "bilinear",
        ImageResizeMode.NearestNeighbor => "nearest_neighbor",
        _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, null),
    };

    public static string ToC(this ImageResizeTransformationMode mode) => mode switch
    {
        ImageResizeTransformationMode.HalfPixel => "half_pixel",
        ImageResizeTransformationMode.PytorchHalfPixel => "pytorch_half_pixel",
        ImageResizeTransformationMode.AlignCorners => "align_corners",
        ImageResizeTransformationMode.Asymmetric => "asymmetric",
        ImageResizeTransformationMode.TFCropAndResize => "tfcrop_and_resize",
        _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, null),
    };

    public static string ToC(this ImageResizeNearestMode mode) => mode switch
    {
        ImageResizeNearestMode.RoundPreferFloor => "round_prefer_floor",
        ImageResizeNearestMode.RoundPreferCeil => "round_prefer_ceil",
        ImageResizeNearestMode.Floor => "floor",
        ImageResizeNearestMode.Ceil => "ceil",
        _ => throw new ArgumentOutOfRangeException(nameof(mode), mode, null),
    };

    public static string ToC(this AttentionCacheKind kind) => kind switch
    {
        AttentionCacheKind.Key => "key",
        AttentionCacheKind.Value => "value",
        _ => throw new ArgumentOutOfRangeException(nameof(kind), kind, null),
    };

    public static string[] ToSlicing(this IEnumerable<string> dims, string[] begins, IRArray<SBP> ndsbp, Placement placement)
    {
        var hstrides = TensorUtilities.GetStrides(placement.Hierarchy.ToArray());
        var splitHierarchy = Enumerable.Range(0, begins.Length).Select(_ => new List<int>()).ToArray();
        var splits = Enumerable.Range(0, begins.Length).Select(_ => new List<int>()).ToArray();
        foreach (var (sbp, i) in ndsbp.Select((s, i) => (s, i)))
        {
            if (sbp is SBPSplit split)
            {
                splits[i] = split.Axes.ToList();
                splitHierarchy[i] = placement.Hierarchy.Select((h, i) => split.Axes.Contains(i) ? h : 1).ToList();
            }
        }

        foreach (var splist in splits)
        {
            splist.Sort((a, b) => -a.CompareTo(b));
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

                begins[i] += " + " + sp.Skip(1).Aggregate($"{placement.Name[sp[0]]}id()", (acc, p) => $"({acc} + {TensorUtilities.GetProduct(splitHierarchy[i].ToArray().AsSpan()[(p + 1)..])} * {placement.Name[p]}id())") + $" * {dimi}";
            }
        }

        return [$"make_ranked_shape({string.Join(',', begins)})", $"fixed_shape<{string.Join(",", dims.Select(d => d.ToString()))}>{{}}"];
    }

    public static string[] ToSlicing(this IEnumerable<string> dims, IRArray<SBP> ndsbp, Placement placement) => ToSlicing(dims, Enumerable.Repeat("0", dims.Count()).ToArray(), ndsbp, placement);

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

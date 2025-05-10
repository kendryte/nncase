// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text.Json;
using System.Text.Json.Serialization;
using Nncase.IR.NN;

namespace Nncase.IO;

public sealed class IPagedAttentionConfigJsonConverter : JsonConverterFactory
{
    public override bool CanConvert(Type typeToConvert)
    {
        return typeof(IPagedAttentionConfig).IsAssignableFrom(typeToConvert);
    }

    public override JsonConverter? CreateConverter(Type typeToConvert, JsonSerializerOptions options)
    {
        return new PagedAttentionConfigConverter();
    }
}

internal sealed class PagedAttentionConfigConverter : JsonConverter<IPagedAttentionConfig>
{
    public override IPagedAttentionConfig? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartObject)
        {
            throw new JsonException("Expected StartObject token");
        }

        using var doc = JsonDocument.ParseValue(ref reader);
        var root = doc.RootElement;

        var numLayers = root.GetProperty(nameof(IPagedAttentionConfig.NumLayers)).GetInt32();
        var numKVHeads = root.GetProperty(nameof(IPagedAttentionConfig.NumKVHeads)).GetInt32();
        var headDim = root.GetProperty(nameof(IPagedAttentionConfig.HeadDim)).GetInt32();
        var blockSize = root.GetProperty(nameof(IPagedAttentionConfig.BlockSize)).GetInt32();
        var kvType = DataType.FromTypeCode((Runtime.TypeCode)root.GetProperty(nameof(IPagedAttentionConfig.KVPrimType)).GetInt32());

        var cacheLayout = JsonSerializer.Deserialize<PagedKVCacheDimKind[]>(
            root.GetProperty(nameof(IPagedAttentionConfig.CacheLayout)).GetRawText(),
            options)!;

        var packedAxes = JsonSerializer.Deserialize<PagedKVCacheDimKind[]>(
            root.GetProperty(nameof(IPagedAttentionConfig.PackedAxes)).GetRawText(),
            options)!;

        var lanes = JsonSerializer.Deserialize<int[]>(
            root.GetProperty(nameof(IPagedAttentionConfig.Lanes)).GetRawText(),
            options)!;

        var shardingAxes = JsonSerializer.Deserialize<PagedKVCacheDimKind[]>(
            root.GetProperty(nameof(IPagedAttentionConfig.ShardingAxes)).GetRawText(),
            options)!;

        var axisPolicies = JsonSerializer.Deserialize<IR.SBPSplit[]>(
            root.GetProperty(nameof(IPagedAttentionConfig.AxisPolicies)).GetRawText(),
            options)!;

        return new PagedAttentionConfig(
            numLayers,
            numKVHeads,
            headDim,
            kvType,
            blockSize,
            cacheLayout,
            packedAxes,
            lanes,
            shardingAxes,
            axisPolicies);
    }

    public override void Write(Utf8JsonWriter writer, IPagedAttentionConfig value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();

        writer.WriteNumber(nameof(IPagedAttentionConfig.NumLayers), value.NumLayers);
        writer.WriteNumber(nameof(IPagedAttentionConfig.NumKVHeads), value.NumKVHeads);
        writer.WriteNumber(nameof(IPagedAttentionConfig.HeadDim), value.HeadDim);
        writer.WriteNumber(nameof(IPagedAttentionConfig.BlockSize), value.BlockSize);

        writer.WritePropertyName("KVType");
        JsonSerializer.Serialize(writer, value.KVPrimType.TypeCode, options);

        writer.WritePropertyName(nameof(IPagedAttentionConfig.CacheLayout));
        JsonSerializer.Serialize(writer, value.CacheLayout, options);

        writer.WritePropertyName(nameof(IPagedAttentionConfig.PackedAxes));
        JsonSerializer.Serialize(writer, value.PackedAxes, options);

        writer.WritePropertyName(nameof(IPagedAttentionConfig.Lanes));
        JsonSerializer.Serialize(writer, value.Lanes, options);

        writer.WritePropertyName(nameof(IPagedAttentionConfig.ShardingAxes));
        JsonSerializer.Serialize(writer, value.ShardingAxes, options);

        writer.WritePropertyName(nameof(IPagedAttentionConfig.AxisPolicies));
        JsonSerializer.Serialize(writer, value.AxisPolicies, options);

        writer.WriteEndObject();
    }
}

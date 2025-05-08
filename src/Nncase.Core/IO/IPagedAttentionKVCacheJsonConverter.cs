// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text.Json;
using System.Text.Json.Serialization;
using Nncase.IR.NN;

namespace Nncase.IO;

public sealed class IPagedAttentionKVCacheJsonConverter : JsonConverterFactory
{
    public override bool CanConvert(Type typeToConvert)
    {
        return typeof(IPagedAttentionKVCache).IsAssignableFrom(typeToConvert);
    }

    public override JsonConverter CreateConverter(Type typeToConvert, JsonSerializerOptions options)
    {
        return new PagedAttentionKVCacheJsonConverter();
    }
}

internal sealed class PagedAttentionKVCacheJsonConverter : JsonConverter<IPagedAttentionKVCache>
{
    public override IPagedAttentionKVCache Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartObject)
        {
            throw new JsonException();
        }

        using var doc = JsonDocument.ParseValue(ref reader);
        var root = doc.RootElement;

        if (!root.TryGetProperty("$type", out var typeProperty))
        {
            throw new JsonException("Missing $type property");
        }

        var typeName = typeProperty.GetString();
        var implementationType = Type.GetType(typeName!)
            ?? throw new JsonException($"Cannot find type {typeName}");

        if (!typeof(IPagedAttentionKVCache).IsAssignableFrom(implementationType))
        {
            throw new JsonException($"Type {typeName} is not assignable to IPagedAttentionKVCache");
        }

        var value = root.GetProperty("Value");
        return (IPagedAttentionKVCache)JsonSerializer.Deserialize(value.GetRawText(), implementationType, options)!;
    }

    public override void Write(Utf8JsonWriter writer, IPagedAttentionKVCache value, JsonSerializerOptions options)
    {
        if (value == null)
        {
            writer.WriteNullValue();
            return;
        }

        var type = value.GetType();
        writer.WriteStartObject();
        writer.WriteString("$type", type.AssemblyQualifiedName);
        writer.WritePropertyName("Value");
        JsonSerializer.Serialize(writer, value, type, options);
        writer.WriteEndObject();
    }
}

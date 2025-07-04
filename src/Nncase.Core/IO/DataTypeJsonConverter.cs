// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Diagnostics.CodeAnalysis;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Nncase.IO;

public sealed class DataTypeJsonConverter : JsonConverterFactory
{
    public override bool CanConvert(Type typeToConvert)
    {
        return typeof(DataType).IsAssignableFrom(typeToConvert);
    }

    public override JsonConverter? CreateConverter(Type typeToConvert, JsonSerializerOptions options)
    {
        return new DataTypeJsonConverterImpl();
    }
}

internal sealed class DataTypeJsonConverterImpl : JsonConverter<DataType>
{
    private const string TypeDiscriminator = "$type";

    public override DataType? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartObject)
        {
            throw new JsonException("Expected StartObject token");
        }

        using var jsonDoc = JsonDocument.ParseValue(ref reader);
        var root = jsonDoc.RootElement;

        if (!root.TryGetProperty(TypeDiscriminator, out var typeProperty))
        {
            throw new JsonException($"Missing {TypeDiscriminator} property");
        }

        var typeName = typeProperty.GetString();
        return typeName switch
        {
            nameof(MaskVectorType) => ReadMaskVectorType(root, options),
            nameof(PrimType) => ReadPrimType(root),
            nameof(PointerType) => ReadPointerType(root, options),
            nameof(ReferenceType) => ReadReferenceType(root, options),
            nameof(VectorType) => ReadVectorType(root, options),
            nameof(ValueType) => ReadValueType(root, options),
            _ => throw new JsonException($"Unknown type discriminator: {typeName}"),
        };
    }

    public override void Write(Utf8JsonWriter writer, DataType value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        var typeToken = value switch
        {
            MaskVectorType => nameof(MaskVectorType),
            PrimType => nameof(PrimType),
            PointerType => nameof(PointerType),
            ReferenceType => nameof(ReferenceType),
            VectorType => nameof(VectorType),
            ValueType => nameof(ValueType),
            _ => throw new JsonException($"Unknown DataType: {value.GetType()}"),
        };
        writer.WriteString(TypeDiscriminator, typeToken);

        switch (value)
        {
            case MaskVectorType vectorType:
                WriteMaskVectorType(writer, vectorType, options);
                break;
            case PrimType primType:
                WritePrimType(writer, primType);
                break;
            case PointerType pointerType:
                WritePointerType(writer, pointerType, options);
                break;
            case ReferenceType referenceType:
                WriteReferenceType(writer, referenceType, options);
                break;
            case VectorType vectorType:
                WriteVectorType(writer, vectorType, options);
                break;
            case ValueType valueType:
                WriteValueType(writer, valueType, options);
                break;
            default:
                throw new JsonException($"Unknown DataType: {value.GetType()}");
        }

        writer.WriteEndObject();
    }

    private static PrimType ReadPrimType(JsonElement element)
    {
        var typeCode = (Runtime.TypeCode)element.GetProperty(nameof(PrimType.TypeCode)).GetInt32();
        return DataType.FromTypeCode(typeCode);
    }

    private static void WritePrimType(Utf8JsonWriter writer, PrimType value)
    {
        writer.WriteNumber(nameof(PrimType.TypeCode), (int)value.TypeCode);
    }

    private static PointerType ReadPointerType(JsonElement element, JsonSerializerOptions options)
    {
        var elemType = JsonSerializer.Deserialize<DataType>(
            element.GetProperty(nameof(PointerType.ElemType)).GetRawText(),
            options) ?? throw new JsonException("Failed to deserialize ElemType");
        return new PointerType(elemType);
    }

    private static void WritePointerType(Utf8JsonWriter writer, PointerType value, JsonSerializerOptions options)
    {
        writer.WritePropertyName(nameof(PointerType.ElemType));
        JsonSerializer.Serialize(writer, value.ElemType, options);
    }

    private static ReferenceType ReadReferenceType(JsonElement element, JsonSerializerOptions options)
    {
        var elemType = JsonSerializer.Deserialize<DataType>(
            element.GetProperty(nameof(ReferenceType.ElemType)).GetRawText(),
            options) ?? throw new JsonException("Failed to deserialize ElemType");
        return new ReferenceType(elemType);
    }

    private static void WriteReferenceType(Utf8JsonWriter writer, ReferenceType value, JsonSerializerOptions options)
    {
        writer.WritePropertyName(nameof(PointerType.ElemType));
        JsonSerializer.Serialize(writer, value.ElemType, options);
    }

    private static MaskVectorType ReadMaskVectorType(JsonElement element, JsonSerializerOptions options)
    {
        var style = JsonSerializer.Deserialize<MaskVectorStyle>(
            element.GetProperty(nameof(MaskVectorType.Style)).GetRawText(),
            options);
        var elementBits = JsonSerializer.Deserialize<int>(
            element.GetProperty(nameof(MaskVectorType.ElementBits)).GetRawText(),
            options);
        var lanes = JsonSerializer.Deserialize<int>(
            element.GetProperty(nameof(MaskVectorType.Lanes)).GetRawText(),
            options);
        return new MaskVectorType(style, elementBits, lanes);
    }

    private static VectorType ReadVectorType(JsonElement element, JsonSerializerOptions options)
    {
        var elemType = JsonSerializer.Deserialize<DataType>(
            element.GetProperty(nameof(VectorType.ElemType)).GetRawText(),
            options) ?? throw new JsonException("Failed to deserialize ElemType");
        var lanes = JsonSerializer.Deserialize<int[]>(
            element.GetProperty(nameof(VectorType.Lanes)).GetRawText(),
            options) ?? throw new JsonException("Failed to deserialize Lanes");
        return new VectorType(elemType, lanes);
    }

    private static void WriteMaskVectorType(Utf8JsonWriter writer, MaskVectorType value, JsonSerializerOptions options)
    {
        writer.WritePropertyName(nameof(MaskVectorType.Style));
        JsonSerializer.Serialize(writer, value.Style, options);
        writer.WritePropertyName(nameof(MaskVectorType.ElementBits));
        JsonSerializer.Serialize(writer, value.ElementBits, options);
        writer.WritePropertyName(nameof(MaskVectorType.Lanes));
        JsonSerializer.Serialize(writer, value.Lanes, options);
    }

    private static void WriteVectorType(Utf8JsonWriter writer, VectorType value, JsonSerializerOptions options)
    {
        writer.WritePropertyName(nameof(VectorType.ElemType));
        JsonSerializer.Serialize(writer, value.ElemType, options);
        writer.WritePropertyName(nameof(VectorType.Lanes));
        JsonSerializer.Serialize(writer, value.Lanes.ToArray(), options);
    }

    private static ValueType ReadValueType(JsonElement element, JsonSerializerOptions options)
    {
        if (!element.TryGetProperty("Name", out var typeProperty))
        {
            throw new JsonException($"Missing {TypeDiscriminator} property");
        }

        var typeName = typeProperty.GetString();
        var bytes = JsonSerializer.Deserialize<int[]>(
            element.GetProperty(nameof(ValueType.Uuid)).GetRawText(),
            options) ?? throw new JsonException("Failed to deserialize Lanes");
        var uuid = new Guid(bytes.Select(i => checked((byte)i)).ToArray());
        return typeName switch
        {
            nameof(IR.NN.AttentionKVCacheType) => new IR.NN.AttentionKVCacheType(),
            nameof(IR.NN.PagedAttentionKVCacheType) => new IR.NN.PagedAttentionKVCacheType(),
            _ => throw new JsonException($"Unknown ValueType discriminator: {typeName}"),
        };
    }

    private static void WriteValueType(Utf8JsonWriter writer, ValueType value, JsonSerializerOptions options)
    {
        writer.WritePropertyName("Name");
        JsonSerializer.Serialize(writer, value.GetType().Name, options);
        writer.WritePropertyName(nameof(ValueType.Uuid));
        writer.WriteStartArray();
        var bytes = value.Uuid.ToByteArray();
        for (int i = 0; i < bytes.Length; i++)
        {
            writer.WriteNumberValue(bytes[i]);
        }

        writer.WriteEndArray();
    }
}

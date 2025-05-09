// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Text.Json;
using System.Text.Json.Serialization;

namespace Nncase.IO;

public sealed class ReferenceJsonConverter : JsonConverterFactory
{
    public override bool CanConvert(Type typeToConvert)
    {
        return typeToConvert.IsGenericType && typeToConvert.GetGenericTypeDefinition() == typeof(Reference<>);
    }

    public override JsonConverter? CreateConverter(Type typeToConvert, JsonSerializerOptions options)
    {
        Type elementType = typeToConvert.GetGenericArguments()[0];
        Type converterType = typeof(ReferenceJsonConverterOfT<>).MakeGenericType(elementType);
        return (JsonConverter)Activator.CreateInstance(converterType)!;
    }

    public sealed class ReferenceJsonConverterOfT<T> : JsonConverter<Reference<T>>
    {
        public override Reference<T> Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            if (reader.TokenType != JsonTokenType.StartObject)
            {
                throw new JsonException();
            }

            using var doc = JsonDocument.ParseValue(ref reader);
            var root = doc.RootElement;
            if (root.TryGetProperty("$type", out var typeProp) && typeProp.GetString() == "Reference")
            {
                var valueProp = root.GetProperty(nameof(Reference<T>.Value));
                var value = JsonSerializer.Deserialize<T>(valueProp.GetRawText(), options)!;
                return new Reference<T>(value);
            }

            throw new JsonException();
        }

        public override void Write(Utf8JsonWriter writer, Reference<T> value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            writer.WritePropertyName("$type");
            writer.WriteStringValue("Reference");
            writer.WritePropertyName(nameof(Reference<T>.Value));
            JsonSerializer.Serialize(writer, value.Value, options);
            writer.WriteEndObject();
        }
    }
}

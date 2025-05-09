// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System.Runtime.InteropServices;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace Nncase.IO;

public sealed class TensorJsonConverterFactory : JsonConverterFactory
{
    public override bool CanConvert(Type typeToConvert)
    {
        if (typeToConvert == typeof(Tensor))
        {
            return true;
        }

        return typeToConvert.IsGenericType && typeToConvert.GetGenericTypeDefinition() == typeof(Tensor<>);
    }

    public override JsonConverter? CreateConverter(Type typeToConvert, JsonSerializerOptions options)
    {
        if (typeToConvert.IsGenericType && typeToConvert.GetGenericTypeDefinition() == typeof(Tensor<>))
        {
            Type elementType = typeToConvert.GetGenericArguments()[0];
            Type converterType = typeof(TensorJsonConverter<>).MakeGenericType(elementType);
            return (JsonConverter)Activator.CreateInstance(converterType)!;
        }

        // 对于基类Tensor，如果实际类型是Tensor<T>，也返回具体的TensorJsonConverter<T>
        if (typeToConvert == typeof(Tensor))
        {
            // TensorBaseJsonConverter只处理真正需要基类转换的情况
            return new TensorBaseJsonConverter();
        }

        throw new JsonException($"Unsupported tensor type: {typeToConvert}");
    }
}

internal sealed class TensorBaseJsonConverter : JsonConverter<Tensor>
{
    public static Tensor<T> CreateTensorByBytes<T>(JsonElement buffer, long[] dimensions, long[] strides)
        where T : struct, IEquatable<T>, System.Numerics.INumberBase<T>
    {
        var bytes = buffer.GetBytesFromBase64();
        var casted = MemoryMarshal.Cast<byte, T>(bytes);
        var memory = new Memory<T>(casted.ToArray());
        return new Tensor<T>(memory, dimensions, strides);
    }

    public static Tensor<T> CreateTensorByObjects<T>(JsonElement buffer, long[] dimensions, long[] strides, JsonSerializerOptions options)
        where T : struct, IEquatable<T>
    {
        var arrayLength = buffer.GetArrayLength();
        var arr = new T[arrayLength];
        for (int i = 0; i < arrayLength; i++)
        {
            arr[i] = JsonSerializer.Deserialize<T>(buffer[i].GetRawText(), options);
        }

        return new Tensor<T>(arr, dimensions, strides);
    }

    public override Tensor? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartObject)
        {
            throw new JsonException();
        }

        using var doc = JsonDocument.ParseValue(ref reader);
        var root = doc.RootElement;

        var elementType = JsonSerializer.Deserialize<DataType>(
            root.GetProperty(nameof(Tensor.ElementType)).GetRawText(),
            options)!;

        var dimensions = JsonSerializer.Deserialize<long[]>(
            root.GetProperty(nameof(Tensor.Dimensions)).GetRawText(),
            options)!;

        var strides = JsonSerializer.Deserialize<long[]>(
            root.GetProperty(nameof(Tensor.Strides)).GetRawText(),
            options)!;

        var clrType = elementType.CLRType;
        var concreteType = typeof(Tensor<>).MakeGenericType(clrType);
        var buffer = root.GetProperty("Buffer");

        if (clrType.GetInterfaces().Any(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(System.Numerics.INumberBase<>)))
        {
            var method = typeof(TensorBaseJsonConverter).GetMethod(nameof(CreateTensorByBytes))!
                .MakeGenericMethod(clrType);
            return (Tensor)method.Invoke(null, new object[] { buffer, dimensions, strides })!;
        }
        else
        {
            var method = typeof(TensorBaseJsonConverter).GetMethod(nameof(CreateTensorByObjects))!
                .MakeGenericMethod(clrType);
            return (Tensor)method.Invoke(null, new object[] { buffer, dimensions, strides, options })!;
        }
    }

    public override void Write(Utf8JsonWriter writer, Tensor value, JsonSerializerOptions options)
    {
        Type runtimeType = value.GetType();

        // 如果是Tensor<T>,创建对应的converter并使用它
        if (runtimeType.IsGenericType && runtimeType.GetGenericTypeDefinition() == typeof(Tensor<>))
        {
            Type elementType = runtimeType.GetGenericArguments()[0];
            Type converterType = typeof(TensorJsonConverter<>).MakeGenericType(elementType);
            var converter = (JsonConverter)Activator.CreateInstance(converterType)!;

            var writeMethod = converter.GetType().GetMethod(nameof(Write))!;
            writeMethod.Invoke(converter, new object[] { writer, value, options });
        }
        else
        {
            throw new JsonException($"Cannot serialize tensor of type {runtimeType}");
        }
    }
}

internal sealed class TensorJsonConverter<T> : JsonConverter<Tensor<T>>
    where T : struct, IEquatable<T>
{
    public override Tensor<T>? Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartObject)
        {
            throw new JsonException();
        }

        using var doc = JsonDocument.ParseValue(ref reader);
        var root = doc.RootElement;

        var elementType = JsonSerializer.Deserialize<DataType>(
            root.GetProperty(nameof(Tensor.ElementType)).GetRawText(),
            options)!;

        var dimensions = JsonSerializer.Deserialize<long[]>(
            root.GetProperty(nameof(Tensor.Dimensions)).GetRawText(),
            options)!;

        var strides = JsonSerializer.Deserialize<long[]>(
            root.GetProperty(nameof(Tensor.Strides)).GetRawText(),
            options)!;

        var clrType = elementType.CLRType;
        var concreteType = typeof(Tensor<>).MakeGenericType(clrType);
        var buffer = root.GetProperty("Buffer");

        if (clrType.GetInterfaces().Any(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(System.Numerics.INumberBase<>)))
        {
            var method = typeof(TensorBaseJsonConverter).GetMethod(nameof(TensorBaseJsonConverter.CreateTensorByBytes))!
                .MakeGenericMethod(clrType);
            return (Tensor<T>)method.Invoke(null, new object[] { buffer, dimensions, strides })!;
        }
        else
        {
            var method = typeof(TensorBaseJsonConverter).GetMethod(nameof(TensorBaseJsonConverter.CreateTensorByObjects))!
                .MakeGenericMethod(clrType);
            return (Tensor<T>)method.Invoke(null, new object[] { buffer, dimensions, strides, options })!;
        }
    }

    public override void Write(Utf8JsonWriter writer, Tensor<T> value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();
        writer.WritePropertyName(nameof(Tensor.ElementType));
        JsonSerializer.Serialize(writer, value.ElementType, options);
        writer.WritePropertyName(nameof(Tensor.Dimensions));
        JsonSerializer.Serialize(writer, value.Dimensions.ToArray(), options);
        writer.WritePropertyName(nameof(Tensor.Strides));
        JsonSerializer.Serialize(writer, value.Strides.ToArray(), options);
        writer.WritePropertyName("Buffer");

        if (typeof(T).GetInterfaces().Any(i => i.IsGenericType && i.GetGenericTypeDefinition() == typeof(System.Numerics.INumberBase<>)))
        {
            writer.WriteBase64StringValue(value.BytesBuffer);
        }
        else
        {
            JsonSerializer.Serialize(writer, value.Buffer, options);
        }

        writer.WriteEndObject();
    }
}

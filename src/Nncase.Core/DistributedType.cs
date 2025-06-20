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
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;
using DryIoc.ImTools;

namespace Nncase.IR;

public enum HierarchyKind : byte
{
    Parallel = 0,
    SMT = 1,
}

[JsonConverter(typeof(SBPConverter))]
public abstract record SBP
{
    public static SBPBroadCast B => SBPBroadCast.Instance;

    public static SBPPartial P(ReduceOp op = ReduceOp.Sum) => new SBPPartial(op);

    public static SBPSplit S(IRArray<int> axes) => new SBPSplit(axes);

    public static SBPSplit S(params int[] axes) => new SBPSplit(axes);
}

public sealed record SBPSplit(IRArray<int> Axes) : SBP
{
    public override string ToString() => $"S({string.Join(",", Axes)})";
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

public class SBPConverter : JsonConverter<SBP>
{
    public override SBP Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
    {
        if (reader.TokenType != JsonTokenType.StartObject)
        {
            throw new JsonException();
        }

        string? typeDiscriminator = null;
        SBPSplit? sbpSplit = null;
        SBPPartial? sbpPartial = null;

        while (reader.Read())
        {
            if (reader.TokenType == JsonTokenType.EndObject)
            {
                break;
            }

            if (reader.TokenType == JsonTokenType.PropertyName)
            {
                string? propertyName = reader.GetString();
                reader.Read(); // Move to property value

                switch (propertyName)
                {
                    case "$type":
                        typeDiscriminator = reader.GetString();
                        break;
                    case "Axes":
                        int[] axes = JsonSerializer.Deserialize<int[]>(ref reader, options)!;
                        var irAxes = new IRArray<int>(axes);
                        if (typeDiscriminator == "S")
                        {
                            sbpSplit = new SBPSplit(irAxes);
                        }
                        else
                        {
                            throw new InvalidDataException("Axes must be used in SBP split");
                        }

                        break;
                    case "Op":
                        ReduceOp partialOp = JsonSerializer.Deserialize<ReduceOp>(ref reader, options);
                        sbpPartial = new SBPPartial(partialOp);
                        break;
                    default:
                        reader.Skip();
                        break;
                }
            }
        }

        switch (typeDiscriminator)
        {
            case "B":
                return SBP.B;
            case "P":
                return sbpPartial!;
            case "S":
                return sbpSplit!;
            default:
                throw new JsonException($"Unknown '$type' discriminator: {typeDiscriminator}");
        }
    }

    public override void Write(Utf8JsonWriter writer, SBP value, JsonSerializerOptions options)
    {
        writer.WriteStartObject();

        if (value is SBPBroadCast)
        {
            writer.WriteString("$type", "B");
        }
        else if (value is SBPPartial partialValue)
        {
            writer.WriteString("$type", "P");
            writer.WriteString("Op", partialValue.Op.ToString());
        }
        else if (value is SBPSplit splitValue)
        {
            writer.WriteString("$type", "S");
            writer.WritePropertyName("Axes");
            JsonSerializer.Serialize(writer, splitValue.Axes.ToArray(), options);
        }
        else
        {
            throw new JsonException($"Unknown SBP type: {value.GetType()}");
        }

        writer.WriteEndObject();
    }
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

public sealed record DistributedType(TensorType TensorType, IRArray<SBP> AxisPolicies, Placement Placement) : IRType
{
    public override string ToString() => $"{TensorType}, ({string.Join(',', AxisPolicies)}), {Placement}";
}

// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Importer.Ncnn;

internal class NcnnTensor
{
    public string Name { get; set; } = string.Empty;

    public Shape ShapeHint { get; set; } = Shape.Unranked;

    public override string ToString() => $"{Name}: {ShapeHint}";
}

internal class NcnnLayer
{
    public NcnnLayer(string type, string name, int bottomCount, int topCount)
    {
        Type = type;
        Name = name;
        Bottoms = new NcnnTensor[bottomCount];
        Tops = new NcnnTensor[topCount];
    }

    public string Type { get; }

    public string Name { get; }

    public NcnnTensor[] Bottoms { get; }

    public NcnnTensor[] Tops { get; }

    public ParamDict ParamDict { get; } = new();

    public override string ToString() => $"[{Type}] {Name}";
}

internal class NcnnModel
{
    public static readonly int ExpectedMagic = 7767517;

    public NcnnModel(int magic, IReadOnlyList<NcnnLayer> layers)
    {
        Magic = magic;
        Layers = layers;
    }

    public int Magic { get; }

    public IReadOnlyList<NcnnLayer> Layers { get; }

    public static NcnnModel ParseFromStream(Stream stream)
    {
        using var reader = new StreamReader(stream, leaveOpen: true);
        if (reader.ReadLine() is not string magicStr)
        {
            throw new InvalidDataException("parse magic failed");
        }

        if (reader.ReadLine()?.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries) is not[var layerCountStr, var blobCountStr])
        {
            throw new InvalidDataException("parse layer_count or blob_count failed");
        }

        var magic = int.Parse(magicStr);
        if (magic != ExpectedMagic)
        {
            throw new InvalidDataException("param is too old, please regenerate");
        }

        var layerCount = int.Parse(layerCountStr);
        var blobCount = int.Parse(blobCountStr);
        if (layerCount <= 0 || blobCount <= 0)
        {
            throw new InvalidDataException("invalid layer_count or blob_count");
        }

        var layers = new NcnnLayer[layerCount];
        foreach (ref var layer in layers.AsSpan())
        {
            var fields = reader.ReadLine()!.Split(' ', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries);
            layer = new NcnnLayer(fields[0], fields[1], int.Parse(fields[2]), int.Parse(fields[3]));

            int cntFieldIndex = 4;
            foreach (ref var bottom in layer.Bottoms.AsSpan())
            {
                bottom = new NcnnTensor { Name = fields[cntFieldIndex++] };
            }

            foreach (ref var top in layer.Tops.AsSpan())
            {
                top = new NcnnTensor { Name = fields[cntFieldIndex++] };
            }

            layer.ParamDict.LoadFrom(fields.AsSpan(cntFieldIndex));
        }

        return new NcnnModel(magic, layers);
    }
}

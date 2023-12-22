// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Nncase.Runtime.Ncnn;

public class NcnnModel
{
    public static readonly int ExpectedMagic = 7767517;

    public NcnnModel()
    {
        Magic = ExpectedMagic;
        ModelInputs = new List<NcnnLayer>();
        Layers = new List<NcnnLayer>();
        MemoryDatas = new List<NcnnLayer>();
    }

    public NcnnModel(int magic, IList<NcnnLayer> layers, IList<NcnnLayer>? modelInputs = null, IList<NcnnLayer>? memoryDatas = null)
    {
        Magic = magic;
        ModelInputs = modelInputs;
        Layers = layers;
        MemoryDatas = memoryDatas;
    }

    public int Magic { get; }
    public IList<NcnnLayer> ModelInputs { get; }
    public IList<NcnnLayer> MemoryDatas { get; }
    public IList<NcnnLayer> Layers { get; }

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

    public void Serialize(TextWriter writer)
    {
        // 1. Magic
        writer.WriteLine(Magic);

        // 2. layer_count & blob_count
        writer.WriteLine($"{Layers.Count + MemoryDatas.Count + ModelInputs.Count} {Layers.Select(x => x.Tops.Length).Sum() + MemoryDatas.Select(x => x.Tops.Length).Sum() + ModelInputs.Select(x => x.Tops.Length).Sum()}");

        // 3. inputs
        foreach (var modelInput in ModelInputs)
        {
            modelInput.Serialize(writer);
        }

        // 4. memorydatas
        foreach (var memoryData in MemoryDatas)
        {
            memoryData.Serialize(writer);
        }

        // 5. layers
        foreach (var layer in Layers)
        {
            layer.Serialize(writer);
        }
    }
}

// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR;

namespace Nncase.Runtime.Ncnn;

public class NcnnTensor
{
    public string Name { get; set; } = string.Empty;

    public Shape ShapeHint { get; set; } = Shape.Unranked;

    public override string ToString() => $"{Name}: {ShapeHint}";
}

public class NcnnLayer
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

    public ParamDict ParamDict { get; set; } = new();

    public override string ToString() => $"[{Type}] {Name}";

    public void Serialize(TextWriter writer)
    {
        writer.Write($"{Type}\t{Name}\t{Bottoms.Length} {Tops.Length} ");

        foreach (var bottom in Bottoms)
        {
            writer.Write($"{bottom.Name} ");
        }

        foreach (var top in Tops)
        {
            writer.Write($"{top.Name} ");
        }

        ParamDict.Serialize(writer);
        writer.WriteLine();
    }
}

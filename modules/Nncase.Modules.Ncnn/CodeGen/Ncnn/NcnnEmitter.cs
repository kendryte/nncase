// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Nncase.IR.Ncnn;
using Nncase.Runtime.Ncnn;

namespace Nncase.CodeGen.Ncnn;

internal class NcnnEmitter
{
    private readonly NcnnModel _model;
    private readonly BinaryWriter _binWriter;

    public NcnnEmitter(BinaryWriter binWriter)
    {
        _model = new NcnnModel();
        _binWriter = binWriter;
    }

    public void SaveParam(Stream paramStream)
    {
        using var sw = new StreamWriter(paramStream, Encoding.ASCII, leaveOpen: true);
        _model.Serialize(sw);
    }

    public void Input(string name) =>
        AddLayer("Input", name, Array.Empty<string>(), new[] { name }, null);

    public void Softmax(string name, string input, int axis) =>
        AddLayer("Softmax", name, new[] { input }, new[] { name }, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = axis }, // axis
            [1] = new ParamValue { Kind = ParamKind.Int, IntValue = 1 }, // fixbug0
        });

    public void Unary(string name, string input, UnaryOperationType opTypes) =>
        AddLayer("UnaryOp", name, new[] { input }, new[] { name }, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = (int)opTypes },
        });

    public void BatchNorm(string name, string input, int channels, float eps, float[] slopeData, float[] meanData, float[] varData, float[] biasData/*, float[] aData, float[] bData*/)
    {
        AddLayer("BatchNorm", name, new[] { input }, new[] { name }, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = channels },
            [1] = new ParamValue { Kind = ParamKind.Float, FloatValue = eps },
        });

        WriteFloatArray(slopeData);
        WriteFloatArray(meanData);
        WriteFloatArray(varData);
        WriteFloatArray(biasData);
    }

    private void AddLayer(string type, string name, string[] bottoms, string[] tops, ParamDict? paramDict = null)
    {
        var layer = new NcnnLayer(type, name, bottoms.Length, tops.Length);
        if (paramDict != null)
        {
            layer.ParamDict = paramDict;
        }

        for (int i = 0; i < bottoms.Length; i++)
        {
            layer.Bottoms[i] = new NcnnTensor { Name = bottoms[i] };
        }

        for (int i = 0; i < tops.Length; i++)
        {
            layer.Tops[i] = new NcnnTensor { Name = tops[i] };
        }

        _model.Layers.Add(layer);
    }

    private void WriteFloatArray(float[] data)
    {
        foreach (float value in data)
        {
            _binWriter.Write(value);
        }

    }
}

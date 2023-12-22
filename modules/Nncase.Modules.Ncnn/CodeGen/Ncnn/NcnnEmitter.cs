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
        // using var sw = new StreamWriter(@"/home/curio/Desktop/param.txt", false, Encoding.UTF8);
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

    public void Binary(string name, string inputA, string inputB, BinaryOperationType opTypes, int lOrR, float[] constInput, int[] constShape)
    {
        var inputList = new[] { inputA, inputB };

        if (constInput != null && constShape != null)
        {
            if (lOrR == 1)
            {
                inputList[0] = name + "_memorydata";
            }
            else
            {
                inputList[1] = name + "_memorydata";
            }

            var paramDict = new ParamDict();
            for (int i = 0; i < constShape.Length; i++)
            {
                paramDict[i] = new ParamValue { Kind = ParamKind.Int, IntValue = constShape[constShape.Length - 1 - i] };
            }

            AddLayer("MemoryData", name + "_memorydata", Array.Empty<string>(), new[] { name + "_memorydata" }, paramDict);

            WriteFloatArray(constInput);
        }

        AddLayer("BinaryOp", name, inputList, new[] { name }, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = (int)opTypes },
        });
    }

    private void AddLayer(string type, string name, string[] bottoms, string[] tops, ParamDict? paramDict = null, int layerType = 1)
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

        switch (type)
        {
            case "Input":
                _model.ModelInputs.Add(layer);
                break;
            case "MemoryData":
                _model.MemoryDatas.Add(layer);
                break;
            default:
                _model.Layers.Add(layer);
                break;
        }

    }

    private void WriteFloatArray(float[] data)
    {
        foreach (float value in data)
        {
            _binWriter.Write(value);
        }
    }
}

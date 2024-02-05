// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Toolkit.HighPerformance;
using Nncase.ArgsStruct;
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

    public void SaveBin()
    {
        using (var fileStream = File.Create(Directory.GetCurrentDirectory() + "/ncnn.bin"))
        {
            _binWriter.BaseStream.Seek(0, SeekOrigin.Begin);
            _binWriter.BaseStream.CopyTo(fileStream);
        }
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

    public void Celu(string name, string input, float alpha) =>
        AddLayer("CELU", name, new[] { input }, new[] { name }, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Float, FloatValue = alpha }, // alpha
        });

    public void Clip(string name, string input, float min, float max) =>
        AddLayer("Clip", name, new[] { input }, new[] { name }, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Float, FloatValue = min }, // min
            [1] = new ParamValue { Kind = ParamKind.Float, FloatValue = max }, // max
        });

    public void Concat(string name, string[] input, int axis)
    {
        AddLayer("Concat", name, input, new[] { name }, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = axis }, // axis
        });
    }

    public void Conv(string name, string input, float[] weightsData, float[] biasData, int numOutput, int kernelW, int kernelH, int dilationW, int dilationH, int strideW, int strideH, int padLeft, int padRight, int padBottom, int padTop, int biasTerm, int weightsDataSize, int int8Flag, int actType, float[] actParams, float padValue, int dynamicFlag)
    {
        AddLayer("Convolution", name, new[] { input }, new[] { name }, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = numOutput },
            [1] = new ParamValue { Kind = ParamKind.Int, IntValue = kernelW },
            [11] = new ParamValue { Kind = ParamKind.Int, IntValue = kernelH },
            [2] = new ParamValue { Kind = ParamKind.Int, IntValue = dilationW },
            [12] = new ParamValue { Kind = ParamKind.Int, IntValue = dilationH },
            [3] = new ParamValue { Kind = ParamKind.Int, IntValue = strideW },
            [13] = new ParamValue { Kind = ParamKind.Int, IntValue = strideH },
            [4] = new ParamValue { Kind = ParamKind.Int, IntValue = padLeft },
            [14] = new ParamValue { Kind = ParamKind.Int, IntValue = padTop },
            [16] = new ParamValue { Kind = ParamKind.Int, IntValue = padRight },
            [15] = new ParamValue { Kind = ParamKind.Int, IntValue = padBottom },
            [5] = new ParamValue { Kind = ParamKind.Int, IntValue = biasTerm },
            [6] = new ParamValue { Kind = ParamKind.Int, IntValue = weightsDataSize },
            [7] = new ParamValue { Kind = ParamKind.Int, IntValue = 1 }, // Group
            [8] = new ParamValue { Kind = ParamKind.Int, IntValue = int8Flag },

            // [9] = new ParamValue { Kind = ParamKind.Int, IntValue = actType },
            // [10] = new ParamValue { Kind = ParamKind.ArrayOfFloat, TensorValue = actParams },
            [18] = new ParamValue { Kind = ParamKind.Float, FloatValue = padValue },
            [19] = new ParamValue { Kind = ParamKind.Int, IntValue = dynamicFlag },
        });
        WriteFloatArray(new float[] { 0 }); // quantize flag [Not exist in ncnn op.md]
        WriteFloatArray(weightsData);
        WriteFloatArray(biasData);
    }

    public void Cumsum(string name, string input, int axis)
    {
        AddLayer("CumulativeSum", name, new[] { input }, new[] { name }, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = axis }, // axis
        });
    }

    public void Elu(string name, string input, float alpha) =>
        AddLayer("ELU", name, new[] { input }, new[] { name }, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Float, FloatValue = alpha }, // alpha
        });

    public void Erf(string name, string input) => AddLayer("Erf", name, new[] { input }, new[] { name }, new ParamDict { });

    public void HardSigmoid(string name, string input, float alpha, float beta) =>
        AddLayer("HardSigmoid", name, new[] { input }, new[] { name }, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Float, FloatValue = alpha }, // alpha
            [1] = new ParamValue { Kind = ParamKind.Float, FloatValue = beta }, // beta
        });

    public void HardSwish(string name, string input, float alpha, float beta) =>
        AddLayer("HardSwish", name, new[] { input }, new[] { name }, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Float, FloatValue = alpha }, // alpha
            [1] = new ParamValue { Kind = ParamKind.Float, FloatValue = beta }, // beta
        });

    public void InstanceNorm(string name, string input, int channels, float eps, int affine, float[] gammaData, float[] betaData)
    {
        AddLayer("InstanceNorm", name, new[] { input }, new[] { name }, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = channels }, // channels
            [1] = new ParamValue { Kind = ParamKind.Float, FloatValue = eps }, // eps
            [2] = new ParamValue { Kind = ParamKind.Int, IntValue = affine }, // affine
        });
        WriteFloatArray(gammaData);
        WriteFloatArray(betaData);
    }

    public void LRN(string name, string input, float alpha, float beta, float bias, int size) =>
        AddLayer("LRN", name, new[] { input }, new[] { name }, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = 0 }, // region_type
            [1] = new ParamValue { Kind = ParamKind.Int, IntValue = size }, // size
            [2] = new ParamValue { Kind = ParamKind.Float, FloatValue = alpha }, // alpha
            [3] = new ParamValue { Kind = ParamKind.Float, FloatValue = beta }, // beta
            [4] = new ParamValue { Kind = ParamKind.Float, FloatValue = bias }, // bias
        });

    public void LSTM(string[] name, string input, int hiddenSize, int weightDataSize, int direction, float[] w, float[] r, float[] b)
    {
        AddLayer("LSTM", name[0], new[] { input }, name, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = hiddenSize },
            [1] = new ParamValue { Kind = ParamKind.Int, IntValue = weightDataSize },
            [2] = new ParamValue { Kind = ParamKind.Int, IntValue = direction },
        });
        WriteFloatArray(new float[] { 0 });
        WriteFloatArray(w);
        WriteFloatArray(new float[] { 0 });
        WriteFloatArray(b);
        WriteFloatArray(new float[] { 0 });
        WriteFloatArray(r);
    }

    public void Padding(string[] name, string input, int top, int bottom, int left, int right, int type, float value, int front, int behind)
    {
        AddLayer("Padding", name[0], new[] { input }, name, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = top },
            [1] = new ParamValue { Kind = ParamKind.Int, IntValue = bottom },
            [2] = new ParamValue { Kind = ParamKind.Int, IntValue = left },
            [3] = new ParamValue { Kind = ParamKind.Int, IntValue = right },
            [4] = new ParamValue { Kind = ParamKind.Int, IntValue = type },
            [5] = new ParamValue { Kind = ParamKind.Float, FloatValue = value },

            // [6] for perChannelPadDataSize.
            [7] = new ParamValue { Kind = ParamKind.Int, IntValue = front },
            [8] = new ParamValue { Kind = ParamKind.Int, IntValue = behind },
        });

        // TODO: confirm padValue is a tensor.
        WriteFloatArray(new float[] { value });
    }

    public void Pooling(string[] name, string input, PoolingArgs poolingArgs)
    {
        AddLayer("Pooling", name[0], new[] { input }, name, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = poolingArgs.PoolingType },
            [1] = new ParamValue { Kind = ParamKind.Int, IntValue = poolingArgs.KernelW },
            [11] = new ParamValue { Kind = ParamKind.Int, IntValue = poolingArgs.KernelH },
            [2] = new ParamValue { Kind = ParamKind.Int, IntValue = poolingArgs.StrideW },
            [12] = new ParamValue { Kind = ParamKind.Int, IntValue = poolingArgs.StrideH },
            [3] = new ParamValue { Kind = ParamKind.Int, IntValue = poolingArgs.PadLeft },
            [14] = new ParamValue { Kind = ParamKind.Int, IntValue = poolingArgs.PadRight },
            [13] = new ParamValue { Kind = ParamKind.Int, IntValue = poolingArgs.PadTop },
            [15] = new ParamValue { Kind = ParamKind.Int, IntValue = poolingArgs.PadBottom },
            [4] = new ParamValue { Kind = ParamKind.Int, IntValue = poolingArgs.GlobalPooling ? 1 : 0 },
            [5] = new ParamValue { Kind = ParamKind.Int, IntValue = poolingArgs.PadMode },
            [6] = new ParamValue { Kind = ParamKind.Int, IntValue = poolingArgs.AvgPoolCountIncludePad ? 1 : 0 },
            [7] = new ParamValue { Kind = ParamKind.Int, IntValue = poolingArgs.AdaptivePooling ? 1 : 0 },

            // [8] = new ParamValue { Kind = ParamKind.Int, IntValue = poolingArgs.OutH },
            // [18] = new ParamValue { Kind = ParamKind.Int, IntValue = poolingArgs.OutH },
        });
    }

    public void PRelu(string[] name, string input, float[] slope)
    {
        AddLayer("PRelu", name[0], new[] { input }, name, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = slope.Length },
        });

        WriteFloatArray(slope);
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

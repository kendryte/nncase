// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using DryIoc.ImTools;
using Microsoft.Toolkit.HighPerformance;
using Nncase.ArgsStruct;
using Nncase.IR;
using Nncase.IR.Ncnn;
using Nncase.Runtime.Ncnn;

namespace Nncase.CodeGen.Ncnn;

internal class NcnnEmitter
{
    private readonly NcnnModel _model;
    private readonly BinaryWriter _binWriter;
    private readonly List<float>? _rData;

    public NcnnEmitter(BinaryWriter binWriter)
    {
        _model = new NcnnModel();
        _binWriter = binWriter;
        _rData = new List<float>();
    }

    public List<float>? GetRData()
    {
        return _rData;
    }

    public void SaveParam(Stream paramStream)
    {
        using var sw = new StreamWriter(paramStream, Encoding.ASCII, leaveOpen: true);
        _model.Serialize(sw);
    }

    public void SaveBin(string dumpPath, uint id)
    {
        using (var fileStream = File.Create(Path.Join(dumpPath, $"/ncnn_{id}.bin")))
        {
            _binWriter.BaseStream.Seek(0, SeekOrigin.Begin);
            _binWriter.BaseStream.CopyTo(fileStream);

            // _binWriter.BaseStream.SetLength(0);
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

    public void Conv(string name, string input, ConvArgs args)
    {
        var actData = new List<float> { args.ActivationParams.Length };
        actData.AddRange(args.ActivationParams);

        var param = new ParamDict();
        param.Add(0, new ParamValue { Kind = ParamKind.Int, IntValue = args.NumOutput });
        param.Add(1, new ParamValue { Kind = ParamKind.Int, IntValue = args.KernelW });
        param.Add(11, new ParamValue { Kind = ParamKind.Int, IntValue = args.KernelH });
        param.Add(2, new ParamValue { Kind = ParamKind.Int, IntValue = args.DilationW });
        param.Add(12, new ParamValue { Kind = ParamKind.Int, IntValue = args.DilationH });
        param.Add(3, new ParamValue { Kind = ParamKind.Int, IntValue = args.StrideW });
        param.Add(13, new ParamValue { Kind = ParamKind.Int, IntValue = args.StrideH });
        param.Add(4, new ParamValue { Kind = ParamKind.Int, IntValue = args.PadLeft });
        param.Add(15, new ParamValue { Kind = ParamKind.Int, IntValue = args.PadRight });
        param.Add(14, new ParamValue { Kind = ParamKind.Int, IntValue = args.PadTop });
        param.Add(16, new ParamValue { Kind = ParamKind.Int, IntValue = args.PadBottom });
        param.Add(18, new ParamValue { Kind = ParamKind.Float, FloatValue = args.PadValue });
        param.Add(5, new ParamValue { Kind = ParamKind.Int, IntValue = args.BiasTerm });
        param.Add(6, new ParamValue { Kind = ParamKind.Int, IntValue = args.WeightDataSize });
        param.Add(7, new ParamValue { Kind = ParamKind.Int, IntValue = args.Groups });
        param.Add(8, new ParamValue { Kind = ParamKind.Int, IntValue = args.Int8ScaleTerm });
        param.Add(9, new ParamValue { Kind = ParamKind.Int, IntValue = args.ActivationType });
        param.Add(-10, new ParamValue { Kind = ParamKind.ArrayOfIntOrFloat, TensorValue = actData.ToArray() });
        param.Add(19, new ParamValue { Kind = ParamKind.Int, IntValue = args.DynamicWeight });

        if (args.Groups > 1)
        {
            AddLayer("ConvolutionDepthWise", name, new[] { input }, new[] { name }, param);
        }
        else
        {
            AddLayer("Convolution", name, new[] { input }, new[] { name }, param);
        }

        WriteFloatArray(new float[] { 0 }); // quantize flag [Not exist in ncnn op.md]
        WriteFloatArray(args.WeightData);
        WriteFloatArray(args.BiasData);
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

    public void LayerNorm(string[] name, string input, int affineSize, float eps, int affine, float[] gammaData, float[] betaData)
    {
        AddLayer("LayerNorm", name[0], new[] { input }, name, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = affineSize }, // affineSize
            [1] = new ParamValue { Kind = ParamKind.Float, FloatValue = eps }, // eps
            [2] = new ParamValue { Kind = ParamKind.Int, IntValue = affine }, // affine
        });
        WriteFloatArray(gammaData);
        WriteFloatArray(betaData);
    }

    public void LRN(string[] name, string input, float alpha, float beta, float bias, int size) =>
        AddLayer("LRN", name[0], new[] { input }, name, new ParamDict
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

    public void PReLU(string[] name, string input, float[] slope)
    {
        AddLayer("PReLU", name[0], new[] { input }, name, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = slope.Length },
        });

        WriteFloatArray(slope);
    }

    public void Reduction(string[] name, string input, ReductionArgs reductionArgs)
    {
        var args = new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = reductionArgs.OpType },
            [1] = new ParamValue { Kind = ParamKind.Int, IntValue = reductionArgs.ReduceAll },
        };

        if (reductionArgs.Axes.Length > 0)
        {
            var axesSizeAndData = new List<long> { reductionArgs.Axes.Length };
            foreach (var item in reductionArgs.Axes)
            {
                axesSizeAndData.Add(item);
            }

            args.Add(-3, new ParamValue { Kind = ParamKind.ArrayOfInt, TensorValue = axesSizeAndData.ToArray() });
        }

        args.Add(4, new ParamValue { Kind = ParamKind.Int, IntValue = reductionArgs.Keepdims });
        args.Add(5, new ParamValue { Kind = ParamKind.Int, IntValue = 1 });
        AddLayer("Reduction", name[0], new[] { input }, name, args);
    }

    public void Reshape(string[] name, string input, int[] newshape)
    {
        var args = new ParamDict();
        List<int> index;
        if (newshape.Length < 4)
        {
            index = Enumerable.Range(0, newshape.Length).ToList();
        }
        else
        {
            index = new List<int> { 0, 1, 11, 2 };
        }

        int i = 0;
        foreach (int item in newshape.Reverse())
        {
            args.Add(index[i], new ParamValue { Kind = ParamKind.Int, IntValue = item });
            i += 1;
        }

        AddLayer("Reshape", name[0], new[] { input }, name, args);
    }

    public void SELU(string[] name, string input, float alpha, float gamma)
    {
        AddLayer("SELU", name[0], new[] { input }, name, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Float, FloatValue = alpha },
            [1] = new ParamValue { Kind = ParamKind.Float, FloatValue = gamma },
        });
    }

    public void Crop(string[] name, string input, CropArgs cropArgs)
    {
        var args = new ParamDict();

        // TODO: if need to fit torch crop, add other args into paramDict.
        if (cropArgs.Axes!.Length > 0)
        {
            var startData = new List<long> { cropArgs.Axes.Length };
            var endData = new List<long> { cropArgs.Axes.Length };
            var axisData = new List<long> { cropArgs.Axes.Length };
            for (int i = 0; i < cropArgs.Axes.Length; i++)
            {
                startData.Add(cropArgs.Starts![i]);
                endData.Add(cropArgs.Ends![i]);
                axisData.Add(cropArgs.Axes![i]);
            }

            args.Add(-9, new ParamValue { Kind = ParamKind.ArrayOfInt, TensorValue = startData.ToArray() });
            args.Add(-10, new ParamValue { Kind = ParamKind.ArrayOfInt, TensorValue = endData.ToArray() });
            args.Add(-11, new ParamValue { Kind = ParamKind.ArrayOfInt, TensorValue = axisData.ToArray() });
        }

        AddLayer("Crop", name[0], new[] { input }, name, args);
    }

    public void Sigmoid(string[] name, string input)
    {
        AddLayer("Sigmoid", name[0], new[] { input }, name);
    }

    public void Softplus(string[] name, string input)
    {
        AddLayer("Softplus", name[0], new[] { input }, name);
    }

    public void Slice(string[] name, string input, int[] slices, int axis)
    {
        var sliceData = new List<int> { slices.Length };
        sliceData.AddRange(slices);
        AddLayer("Slice", name[0], new[] { input }, name, new ParamDict
        {
            [-0] = new ParamValue { Kind = ParamKind.ArrayOfInt, TensorValue = sliceData.ToArray() },
            [1] = new ParamValue { Kind = ParamKind.Int, IntValue = axis },
        });
    }

    public void Tile(string[] name, string input, int[] repeats)
    {
        var repeatsData = new List<int> { repeats.Length };
        repeatsData.AddRange(repeats);
        AddLayer("Tile", name[0], new[] { input }, name, new ParamDict
        {
            [-2] = new ParamValue { Kind = ParamKind.ArrayOfInt, TensorValue = repeatsData.ToArray() },
        });
    }

    public void Permute(string[] name, string input, int orderType)
    {
        AddLayer("Permute", name[0], new[] { input }, name, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = orderType },
        });
    }

    public void Matmul(string[] name, string inputA, string inputB, int lOrR, float[] constInput, int[] constShape)
    {
        var inputList = new[] { inputA, inputB };

        if (constInput != null && constShape != null)
        {
            if (lOrR == 1)
            {
                inputList[0] = name[0] + "_memorydata";
            }
            else
            {
                inputList[1] = name[0] + "_memorydata";
            }

            var paramDict = new ParamDict();
            for (int i = 0; i < constShape.Length; i++)
            {
                int index = i switch
                {
                    0 => 0,
                    1 => 1,
                    2 when constShape.Length == 3 => 2,
                    2 when constShape.Length == 4 => 11,
                    3 when constShape.Length == 4 => 2,
                    _ => throw new NotSupportedException("Only support less than 5D"),
                };
                paramDict[index] = new ParamValue { Kind = ParamKind.Int, IntValue = constShape[constShape.Length - 1 - i] };
            }

            AddLayer("MemoryData", name[0] + "_memorydata", Array.Empty<string>(), new[] { name[0] + "_memorydata" }, paramDict);

            WriteFloatArray(constInput);
        }

        AddLayer("MatMul", name[0], inputList, name, null);
    }

    public void ConvTranspose(string[] name, string input, ConvTransposeArgs args)
    {
        var actData = new List<float> { args.ActivationParams.Length };
        actData.AddRange(args.ActivationParams);

        AddLayer("Deconvolution", name[0], new[] { input }, name, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = args.NumOutput },
            [1] = new ParamValue { Kind = ParamKind.Int, IntValue = args.KernelW },
            [11] = new ParamValue { Kind = ParamKind.Int, IntValue = args.KernelH },
            [2] = new ParamValue { Kind = ParamKind.Int, IntValue = args.DilationW },
            [12] = new ParamValue { Kind = ParamKind.Int, IntValue = args.DilationH },
            [3] = new ParamValue { Kind = ParamKind.Int, IntValue = args.StrideW },
            [13] = new ParamValue { Kind = ParamKind.Int, IntValue = args.StrideH },
            [4] = new ParamValue { Kind = ParamKind.Int, IntValue = args.PadLeft },
            [14] = new ParamValue { Kind = ParamKind.Int, IntValue = args.PadTop },
            [15] = new ParamValue { Kind = ParamKind.Int, IntValue = args.PadRight },
            [16] = new ParamValue { Kind = ParamKind.Int, IntValue = args.PadBottom },

            [5] = new ParamValue { Kind = ParamKind.Int, IntValue = args.BiasTerm },
            [6] = new ParamValue { Kind = ParamKind.Int, IntValue = args.WeightDataSize },

            // [9] = new ParamValue { Kind = ParamKind.Int, IntValue = args.ActivationType },
            // [-10] = new ParamValue { Kind = ParamKind.ArrayOfIntOrFloat, TensorValue = actData.ToArray() },
            //
            // [18] = new ParamValue { Kind = ParamKind.Int, IntValue = args.OutputPadRight },
            // [19] = new ParamValue { Kind = ParamKind.Int, IntValue = args.OutputPadBottom },
            // [20] = new ParamValue { Kind = ParamKind.Int, IntValue = args.OutputW },
            // [21] = new ParamValue { Kind = ParamKind.Int, IntValue = args.OutputH },
        });
        WriteFloatArray(new float[] { 0 }); // quantize flag [Not exist in ncnn op.md]
        WriteFloatArray(args.WeightData.ToArray<float>());
        WriteFloatArray(args.BiasData);
    }

    public void Cast(string[] name, string input, int fromType, int toType)
    {
        AddLayer("Cast", name[0], new[] { input }, name, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = fromType },
            [1] = new ParamValue { Kind = ParamKind.Int, IntValue = toType },
        });
    }

    public void GELU(string[] name, string input)
    {
        AddLayer("GELU", name[0], new[] { input }, name, null);
    }

    public void Dequantize(string[] name, string input, float[] scale, float[] bias)
    {
        AddLayer("Dequantize", name[0], new[] { input }, name, new ParamDict
        {
            [0] = new ParamValue { Kind = ParamKind.Int, IntValue = scale.Length },
            [1] = new ParamValue { Kind = ParamKind.Int, IntValue = bias.Length },
        });

        WriteFloatArray(scale);
        WriteFloatArray(bias);
    }

    public void Squeeze(string[] name, string input, int[] dims)
    {
        var repeatsData = new List<int> { dims.Length };
        repeatsData.AddRange(dims);
        var args = new ParamDict();
        if (dims.Length == 0)
        {
            args.Add(0, new ParamValue { Kind = ParamKind.Int, IntValue = 1 });
            args.Add(1, new ParamValue { Kind = ParamKind.Int, IntValue = 1 });
            args.Add(2, new ParamValue { Kind = ParamKind.Int, IntValue = 1 });
        }
        else
        {
            args.Add(-3, new ParamValue { Kind = ParamKind.ArrayOfInt, TensorValue = repeatsData.ToArray() });
        }

        AddLayer("Squeeze", name[0], new[] { input }, name, args);
    }

    public void Unsqueeze(string[] name, string input, int[] dims)
    {
        var repeatsData = new List<int> { dims.Length };
        repeatsData.AddRange(dims);
        AddLayer("ExpandDims", name[0], new[] { input }, name, new ParamDict
        {
            [-3] = new ParamValue { Kind = ParamKind.ArrayOfInt, TensorValue = repeatsData.ToArray() },
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
        _rData!.AddRange(data);
        foreach (float value in data)
        {
            _binWriter.Write(value);
        }
    }
}

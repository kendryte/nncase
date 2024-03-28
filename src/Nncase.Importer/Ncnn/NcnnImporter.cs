// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using FlatBuffers;
using LanguageExt;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.F;
using Nncase.IR.Math;
using Nncase.IR.NN;
using Nncase.IR.Tensors;
using Nncase.Runtime.Ncnn;
using Math = System.Math;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Importer.Ncnn;

/// <summary>
/// Ncnn importer.
/// </summary>
public sealed partial class NcnnImporter : BaseImporter
{
    private static readonly Dictionary<tflite.TensorType, DataType> _typeMap = new()
    {
        { tflite.TensorType.BOOL, DataTypes.Boolean },
        { tflite.TensorType.FLOAT16, DataTypes.Float16 },
        { tflite.TensorType.FLOAT32, DataTypes.Float32 },
        { tflite.TensorType.FLOAT64, DataTypes.Float64 },
        { tflite.TensorType.INT16, DataTypes.Int16 },
        { tflite.TensorType.INT32, DataTypes.Int32 },
        { tflite.TensorType.INT64, DataTypes.Int64 },
        { tflite.TensorType.INT8, DataTypes.Int8 },
        { tflite.TensorType.STRING, DataTypes.Utf8Char },
        { tflite.TensorType.UINT32, DataTypes.UInt32 },
        { tflite.TensorType.UINT64, DataTypes.UInt64 },
        { tflite.TensorType.UINT8, DataTypes.UInt8 },
    };

    private readonly NcnnModel _model;
    private readonly NcnnModelBin _modelBin;
    private readonly Dictionary<string, Expr> _outputTensors = new Dictionary<string, Expr>();

    /// <summary>
    /// Initializes a new instance of the <see cref="NcnnImporter"/> class.
    /// </summary>
    /// <param name="ncnnParam">Ncnn param stream.</param>
    /// <param name="ncnnBin">Ncnn bin stream.</param>
    /// <param name="compileSession">Compile session.</param>
    public NcnnImporter(Stream ncnnParam, Stream ncnnBin, CompileSession compileSession)
        : base(compileSession)
    {
        _model = NcnnModel.ParseFromStream(ncnnParam);
        _modelBin = new NcnnModelBin(ncnnBin);
    }

    /// <inheritdoc/>
    protected override (IEnumerable<Var> Inputs, Dictionary<Var, Expr[]> VarMap) CreateInputs()
    {
        var inputs = new List<Var>();
        var varMap = new Dictionary<Var, Expr[]>();

        foreach (var layer in _model.Layers.Where(x => x.Type == "Input"))
        {
            var input = new Var(layer.Name, TensorType.Unranked(DataTypes.Float32));
            inputs.Add(input);
            _outputTensors.Add(layer.Name, input);
        }

        return (inputs, varMap);
    }

    protected override void ConvertOp()
    {
        foreach (var layer in _model.Layers.Where(x => x.Type != "Input"))
        {
            Visit(layer);
        }
    }

    protected override Expr CreateOutputs()
    {
        var outputTensors = (from l in _model.Layers
                             from t in l.Tops
                             select t).Select((x, i) => (x.Name, i)).ToDictionary(x => x.Name, x => x.i);
        var unusedTensors = (from t in outputTensors.Keys.Except(from l in _model.Layers
                                                                 from t in l.Bottoms
                                                                 select t.Name)
                             orderby outputTensors[t]
                             select t).ToArray();
        var outputs = unusedTensors.Select(x => _outputTensors[x]).ToArray();
        var body = outputs.Length > 1 ? new IR.Tuple(outputs) : outputs[0];
        return body;
    }

    private static Expr CHWToNCHW(Expr expr) =>
        Tensors.Unsqueeze(expr, new[] { 0 });

    private static Expr NCHWToCHW(Expr expr) =>
        Tensors.Squeeze(expr, new[] { 0 });

    private static ValueRange<float> ToFloatClampRange(int activationType, ReadOnlySpan<float> activationParams) =>
        activationType switch
        {
            1 => new(0, float.PositiveInfinity),
            3 => new(activationParams[0], activationParams[1]),
            _ => ValueRange<float>.Full,
        };

    private static Expr ApplyActivation(Expr input, int activationType, ReadOnlySpan<float> activationParams) =>
        activationType switch
        {
            0 => input,
            1 => NN.Relu(input),
            2 => NN.LeakyRelu(input, activationParams[0]),
            3 => IR.F.Math.Clamp(input, activationParams[0], activationParams[1]),
            4 => NN.Sigmoid(input),
            5 => NN.Mish(input),
            _ => throw new NotSupportedException($"Unsupported activation type: {activationType}."),
        };

    private void Visit(NcnnLayer layer)
    {
        var output = layer.Type switch
        {
            "Concat" => VisitConcat(layer),
            "Convolution" => VisitConvolution(layer),
            "ConvolutionDepthWise" => VisitConvolution(layer),
            "InnerProduct" => VisitInnerProduct(layer),
            "Pooling" => VisitPooling(layer),
            "ShuffleChannel" => VisitShuffleChannel(layer),
            "Softmax" => VisitSoftmax(layer),
            "Split" => VisitSplit(layer),
            _ => UnSupportedOp(layer.Type),
        };

        var outputNames = layer.Tops.Select(x => x.Name).ToArray();
        output.Metadata.OutputNames = outputNames;
        AddToOutputs(_outputTensors, outputNames, output);
    }

    private Expr GetInputExprs(NcnnLayer layer, int index) =>
        _outputTensors[layer.Bottoms[index].Name];

    private (Expr Expr0, Expr Expr1) GetInputExprs(NcnnLayer layer, int index0, int index1) =>
        (GetInputExprs(layer, index0), GetInputExprs(layer, index1));

    private IEnumerable<Expr> GetInputExprs(NcnnLayer layer) =>
        layer.Bottoms.Select(x => _outputTensors[x.Name]);
}

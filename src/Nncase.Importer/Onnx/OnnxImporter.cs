// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Nncase.IR;
using Google.Protobuf;
using Google.Protobuf.Collections;
using LanguageExt;
using Nncase.IR.Tensors;
using Onnx;
using static Nncase.IR.F.Tensors;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Importer;

public sealed partial class OnnxImporter : BaseImporter
{

    private readonly ModelProto _model;
    private readonly GraphProto _graph;
    private readonly Dictionary<string, long> _opSetMap;
    private Dictionary<string, Expr> _outputTensors;
    private Dictionary<string, TensorProto> _constTensors;

    public OnnxImporter(byte[] onnxModel)
    {
        _opSetMap = new Dictionary<string, long>();
        var m = new MessageParser<ModelProto>(
            () => new ModelProto());

        // todo:how to check valid?
        _model = m.ParseFrom(onnxModel);
        foreach (var opSet in _model.OpsetImport)
        {
            _opSetMap.Add(opSet.Domain, opSet.Version);
        }

        _graph = _model.Graph;
    }


    public override IEnumerable<Var> CreateInputs()
    {
        _constTensors = _graph.Initializer
            .ToDictionary(tensor => tensor.Name, tensor => tensor);

        var createdInputs = _graph.Input
            .Where(n => !_constTensors.ContainsKey(n.Name))
            .Select(n => new Var(n.Name, GetIRType(n)));

        _outputTensors = createdInputs.ToDictionary(n => n.Name, n => (Expr) n);
        return createdInputs;
    }

    public override void ConvertOp()
    {
        _graph.Node.ToList().ForEach(Visit);
    }

    public override Expr CreateOutputs()
    {
        var outputs = _graph.Output.Select(o => _outputTensors[o.Name]).ToArray();
        var body = outputs.Length > 1 ? new IR.Tuple(outputs) : outputs[0];
        return body;
    }

    private void Visit(NodeProto op)
    {
        // Console.WriteLine(op.OpType);
        _opsInModel.Add(op.OpType);
        var output = op.OpType switch
        {
            "Abs" => VisitUnary(op, UnaryOp.Abs),
            "Acos" => VisitUnary(op, UnaryOp.Acos),
            "Acosh" => VisitUnary(op, UnaryOp.Acosh),
            "And" => VisitBinary(op, BinaryOp.LogicalAnd),
            "ArgMax" => VisitReduceArg(op, ReduceArgOp.ArgMax),
            "ArgMin" => VisitReduceArg(op, ReduceArgOp.ArgMin),
            "Asin" => VisitUnary(op, UnaryOp.Asin),
            "Asinh" => VisitUnary(op, UnaryOp.Asinh),
            "Add" => VisitBinary(op, BinaryOp.Add),
            "AveragePool" => VisitReduceWindow2D(op, ReduceOp.Mean, 0f),
            "BatchNormalization" => VisitBatchNormalization(op),
            "Cast" => VisitCast(op),
            "Ceil" => VisitUnary(op, UnaryOp.Ceil),
            "Celu" => VisitCelu(op),
            "Clip" => VisitClip(op),
            "Concat" => VisitConcat(op),
            "Constant" => VisitConstant(op),
            "ConstantOfShape" => VisitConstantOfShape(op),
            "Conv" => VisitConv2D(op),
            "ConvTranspose" => VisitConv2DTranspose(op),
            "Cos" => VisitUnary(op, UnaryOp.Cos),
            "Cosh" => VisitUnary(op, UnaryOp.Cosh),
            "CumSum" => VisitCumSum(op),
            "DepthToSpace" => VisitDepthToSpace(op),

            "DequantizeLinear" => VisitDequantizeLinear(op),
            "Div" => VisitBinary(op, BinaryOp.Div),
            "Dropout" => VisitDropout(op),
            "Elu" => VisitElu(op),
            "Equal" => VisitCompare(op, CompareOp.Equal),
            "Exp" => VisitUnary(op, UnaryOp.Exp),
            "Expand" => VisitExpand(op),
            "Flatten" => VisitFlatten(op),
            "Floor" => VisitUnary(op, UnaryOp.Floor),
            "Gather" => VisitGather(op),
            "GatherND" => VisitGatherND(op),
            "Gemm" => VisitGemm(op),
            "GlobalAveragePool" => VisitReduceWindow2D(op, ReduceOp.Mean, 0f, true),
            "GlobalMaxPool" => VisitReduceWindow2D(op, ReduceOp.Max, float.MinValue, true),
            "Hardmax" => VisitHardmax(op),
            "HardSigmoid" => VisitHardSigmoid(op),
            "HardSwish" => VisitHardSwish(op),
            "Identity" => VisitIdentity(op),
            "InstanceNormalization" => VisitInstanceNormalization(op),
            "LpNormalization" => VisitLpNormalization(op),
            "LeakyRelu" => VisitLeakyRelu(op),
            "Less" => VisitCompare(op, CompareOp.LowerThan),
            "Log" => VisitUnary(op, UnaryOp.Log),
            "LogSoftmax" => VisitLogSoftmax(op),
            "LRN" => VisitLRN(op),
            "LSTM" => VisitLSTM(op),
            "MatMul" => VisitMatMul(op),
            "MaxPool" => VisitReduceWindow2D(op, ReduceOp.Max, float.MinValue),
            "Max" => VisitBinary(op, BinaryOp.Max),
            "Min" => VisitBinary(op, BinaryOp.Min),
            "Mul" => VisitBinary(op, BinaryOp.Mul),
            "Neg" => VisitUnary(op, UnaryOp.Neg),
            "Not" => VisitUnary(op, UnaryOp.LogicalNot),
            "OneHot" => VisitOneHot(op),
            "Pad" => VisitPad(op),
            "Pow" => VisitBinary(op, BinaryOp.Pow),
            "PRelu" => VisitPRelu(op),

            "QuantizeLinear" => VisitQuantizeLinear(op),
            "RandomNormal" => VisitRandomNormal(op),
            "RandomNormalLike" => VisitRandomNormalLike(op),
            "RandomUniform" => VisitRandomUniform(op),
            "RandomUniformLike" => VisitRandomUniformLike(op),
            "Range" => VisitRange(op),
            "ReduceL1" => VisitReduceL1(op),
            "ReduceL2" => VisitReduceL2(op),
            "ReduceLogSum" => VisitReduceLogSum(op),
            "ReduceLogSumExp" => VisitReduceLogSumExp(op),
            "ReduceMax" => VisitReduce(op, ReduceOp.Max, float.MinValue),
            "ReduceMean" => VisitReduce(op, ReduceOp.Mean, 0f),
            "ReduceMin" => VisitReduce(op, ReduceOp.Min, float.MaxValue),
            "ReduceProd" => VisitReduce(op, ReduceOp.Prod, 1f),
            "ReduceSum" => VisitReduce(op, ReduceOp.Sum, 0f),
            "ReduceSumSquare" => VisitReduceSumSquare(op),
            "Relu" => VisitRelu(op),
            "Reshape" => VisitReshape(op),

            "Resize" => VisitResize(op),
            "ReverseSequence" => VisitReverseSequence(op),
            "Round" => VisitUnary(op, UnaryOp.Round),
            "Selu" => VisitSelu(op),
            "Shape" => VisitShape(op),
            "Sin" => VisitUnary(op, UnaryOp.Sin),
            "Sinh" => VisitUnary(op, UnaryOp.Sinh),
            "Sigmoid" => VisitSigmoid(op),
            "Sign" => VisitUnary(op, UnaryOp.Sign),
            "Size" => VisitSize(op),
            "Slice" => VisitSlice(op),
            "Softmax" => VisitSoftmax(op),
            "Softplus" => VisitSoftplus(op),
            "Softsign" => VisitSoftsign(op),
            "SpaceToDepth" => VisitSpaceToDepth(op),
            "Split" => VisitSplit(op),
            "Sqrt" => VisitUnary(op, UnaryOp.Sqrt),
            "Squeeze" => VisitSqueeze(op),
            "Sub" => VisitBinary(op, BinaryOp.Sub),
            "Sum" => VisitSum(op),
            "Tanh" => VisitUnary(op, UnaryOp.Tanh),
            "Tile" => VisitTile(op),
            "Transpose" => VisitTranspose(op),

            // "Upsample" => VisitUpsample(op),
            "Unsqueeze" => VisitUnsqueeze(op),
            "Where" => VisitWhere(op),
            _ => UnSupportedOp(op.OpType)
        };
        AddToOutputs(_outputTensors, op.Output.ToArray(), output);
    }

    // about op set: https://github.com/onnx/onnx/issues/3678
    private long GetOpSet(string domain)
    {
        return _opSetMap[domain];
    }

    private long GetOpSet(NodeProto node)
    {
        return _opSetMap[node.Domain];
    }
}

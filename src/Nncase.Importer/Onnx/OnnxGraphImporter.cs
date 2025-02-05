// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Google.Protobuf;
using Google.Protobuf.Collections;
using LanguageExt;
using NetFabric.Hyperlinq;
using Nncase.IR;
using Nncase.IR.Tensors;
using Onnx;
using Tuple = Nncase.IR.Tuple;

namespace Nncase.Importer;

public sealed record OnnxParentGraphInfo(IReadOnlyDictionary<string, long> OpSetMap, IReadOnlyDictionary<string, Expr> OutputTensors);

public sealed partial class OnnxGraphImporter : BaseGraphImporter
{
    private readonly OnnxGraphImporter? _parent;
    private readonly GraphProto _graph;
    private readonly IReadOnlyDictionary<string, long> _opSetMap;
    private Dictionary<string, TensorProto>? _constTensors;
    private Dictionary<string, Var> _dynVarMap = new();
    private Dictionary<string, Expr> _outputTensors = new();

    /// <summary>
    /// Initializes a new instance of the <see cref="OnnxGraphImporter"/> class.
    /// </summary>
    /// <param name="parent">Parent graph.</param>
    /// <param name="graph">Onnx graph.</param>
    /// <param name="opSetMap">Op set map.</param>
    /// <param name="compileSession">Compile session.</param>
    /// <param name="module">IRModule.</param>
    public OnnxGraphImporter(OnnxGraphImporter? parent, GraphProto graph, IReadOnlyDictionary<string, long> opSetMap, CompileSession compileSession, IRModule module)
        : base(graph.Name, compileSession, module)
    {
        _parent = parent;
        _graph = graph;
        _opSetMap = opSetMap;
    }

    public (IEnumerable<Var> Inputs, Dictionary<Var, Expr[]> VarMap) CreateModelInputs(ModelProto model)
    {
        var graph = model.Graph;
        var bucketOptions = CompileSession.CompileOptions.ShapeBucketOptions;
        _constTensors = graph.Initializer
            .ToDictionary(tensor => tensor.Name, tensor => tensor);

        var originInputs = graph.Input
            .Where(n => !_constTensors.ContainsKey(n.Name));
        Inputs = originInputs.Select(n => new Var(n.Name, GetIRType(n))).ToList();
        var dynVarNames = (from input in graph.Input
                           from dim in input.Type.TensorType.Shape.Dim.Select((dim, i) => (dim, i))
                           where IsDynamicDim(dim.dim)
                           select GetDimParam(input.Name, dim.dim, dim.i)).ToHashSet();
        _dynVarMap = dynVarNames.Select(v => new Var(v, new TensorType(DataTypes.Int32, Shape.Scalar)))
            .ToDictionary(v => v.Name, v => v);
        var varMap = originInputs
            .Select((v, i) => (Inputs[i], GetOriginShape(v)))
            .ToDictionary(tup => tup.Item1, tup => tup.Item2);

        CompileSession.CompileOptions.ShapeBucketOptions =
            bucketOptions with { VarMap = varMap };
        _outputTensors = Inputs.ToDictionary(n => n.Name, n =>
        {
            if (bucketOptions.FixVarMap.TryGetValue(n.Name, out var fixDim))
            {
                return (long)fixDim;
            }
            else
            {
                return (Expr)n;
            }
        });
        return (Inputs, varMap);
    }

    /// <inheritdoc/>
    protected override void ConvertOp()
    {
        foreach (var node in _graph.Node)
        {
            Visit(node);
        }
    }

    /// <inheritdoc/>
    protected override Expr CreateOutputs()
    {
        var outputs = _graph.Output.Select(o => _outputTensors![o.Name]).ToArray();
        var body = outputs.Length > 1 ? new IR.Tuple(outputs) : outputs[0];
        return body;
    }

    private void Visit(NodeProto op)
    {
        AddOpInModel(op.OpType);
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
            "Einsum" => VisitEinsum(op),
            "Elu" => VisitElu(op),
            "Equal" => VisitCompare(op, CompareOp.Equal),
            "Exp" => VisitUnary(op, UnaryOp.Exp),
            "Erf" => VisitErf(op),
            "Expand" => VisitExpand(op),
            "Flatten" => VisitFlatten(op),
            "Floor" => VisitUnary(op, UnaryOp.Floor),
            "Gather" => VisitGather(op),
            "GatherElements" => VisitGatherElements(op),
            "GatherND" => VisitGatherND(op),
            "Gemm" => VisitGemm(op),
            "GlobalAveragePool" => VisitReduceWindow2D(op, ReduceOp.Mean, 0f, true),
            "GlobalMaxPool" => VisitReduceWindow2D(op, ReduceOp.Max, float.MinValue, true),
            "Greater" => VisitCompare(op, CompareOp.GreaterThan),
            "GreaterOrEqual" => VisitCompare(op, CompareOp.GreaterOrEqual),
            "Hardmax" => VisitHardmax(op),
            "HardSigmoid" => VisitHardSigmoid(op),
            "HardSwish" => VisitHardSwish(op),
            "Identity" => VisitIdentity(op),
            "If" => VisitIf(op),
            "InstanceNormalization" => VisitInstanceNormalization(op),
            "LayerNormalization" => VisitLayerNormalization(op),
            "LpNormalization" => VisitLpNormalization(op),
            "LeakyRelu" => VisitLeakyRelu(op),
            "Less" => VisitCompare(op, CompareOp.LowerThan),
            "LessOrEqual" => VisitCompare(op, CompareOp.LowerOrEqual),
            "Log" => VisitUnary(op, UnaryOp.Log),
            "LogSoftmax" => VisitLogSoftmax(op),
            "LRN" => VisitLRN(op),
            "LSTM" => VisitLSTM(op),
            "MatMul" => VisitMatMul(op),
            "MaxPool" => VisitReduceWindow2D(op, ReduceOp.Max, float.MinValue),
            "Max" => VisitBinary(op, BinaryOp.Max),
            "Min" => VisitBinary(op, BinaryOp.Min),
            "Mod" => VisitBinary(op, BinaryOp.Mod),
            "Mul" => VisitBinary(op, BinaryOp.Mul),
            "Neg" => VisitUnary(op, UnaryOp.Neg),
            "Not" => VisitUnary(op, UnaryOp.LogicalNot),
            "Or" => VisitBinary(op, BinaryOp.LogicalOr),
            "OneHot" => VisitOneHot(op),
            "Pad" => VisitPad(op),
            "Pow" => VisitBinary(op, BinaryOp.Pow),
            "PRelu" => VisitPRelu(op),

            "QuantizeLinear" => VisitQuantizeLinear(op),
            "QLinearConv" => VisitQLinearConv(op),
            "QLinearMatmul" => VisitQLinearMatMul(op),
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
            "ScatterND" => VisitScatterND(op),
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
            "Trilu" => VisitTrilu(op),
            "TopK" => VisitTopK(op),
            "Transpose" => VisitTranspose(op),

            "Upsample" => VisitUpsample(op),
            "Unsqueeze" => VisitUnsqueeze(op),
            "Where" => VisitWhere(op),
            _ => UnSupportedOp(op.OpType),
        };

        List<string> outputNames = new();

        var outputsCount = op.Output.Count;
        for (int i = 0; i < outputsCount; i++)
        {
            outputNames.Add(op.Output[i]);
        }

        output.Metadata.OutputNames = outputNames;

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

    private OnnxGraphImporter CreateSubgraphImporter(GraphProto graph)
    {
        return new OnnxGraphImporter(this, graph, _opSetMap, CompileSession, IRModule);
    }
}

using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Nncase.IR;
using Google.Protobuf;
using Google.Protobuf.Collections;
using NetFabric.Hyperlinq;
using Nncase.IR.Math;
using Onnx;
using static Google.Protobuf.MessageParser;

namespace Nncase.Importer
{
    public sealed partial class OnnxImporter
    {
        private static readonly Dictionary<TensorProto.Types.DataType, DataType> _typeMap = new()
        {
            { TensorProto.Types.DataType.Bool, DataType.Bool },
            { TensorProto.Types.DataType.Float16, DataType.Float16 },
            { TensorProto.Types.DataType.Float, DataType.Float32 },
            { TensorProto.Types.DataType.Double, DataType.Float64 },
            { TensorProto.Types.DataType.Int16, DataType.Int16 },
            { TensorProto.Types.DataType.Int32, DataType.Int32 },
            { TensorProto.Types.DataType.Int64, DataType.Int64 },
            { TensorProto.Types.DataType.Int8, DataType.Int8 },
            { TensorProto.Types.DataType.String, DataType.String },
            { TensorProto.Types.DataType.Uint32, DataType.UInt32 },
            { TensorProto.Types.DataType.Uint64, DataType.UInt64 },
            { TensorProto.Types.DataType.Uint8, DataType.UInt8 },
        };

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

        private Const GetConst(TensorProto tensor)
        {
            var shape = GetShape(tensor);
            var type = GetDataType(tensor);
            var dt = (TensorProto.Types.DataType)tensor.DataType;
            // should not use tensor.DataLocation to distinguish whether it is RawData
            if (tensor.RawData.ToByteArray().Length() != 0)
            {
                return new Const(new TensorType(type, shape), tensor.RawData.ToByteArray());
            }
            else
            {
                return dt switch
                {
                    // todo:not directly supported type should convert
                    //TensorProto.Types.DataType.Bool => Const.FromSpan(),
                    //TensorProto.Types.DataType.Float16 => Const.FromSpan(),
                    TensorProto.Types.DataType.Float => Const.FromSpan<float>(tensor.FloatData.ToArray(), shape),
                    TensorProto.Types.DataType.Double => Const.FromSpan<double>(tensor.DoubleData.ToArray(), shape),
                    //TensorProto.Types.DataType.Int16 => Const.FromSpan(),
                    TensorProto.Types.DataType.Int32 => Const.FromSpan<int>(tensor.Int32Data.ToArray(), shape),
                    TensorProto.Types.DataType.Int64 => Const.FromSpan<long>(tensor.Int64Data.ToArray(), shape),
                    //TensorProto.Types.DataType.Int8 => Const.FromSpan(),
                    //TensorProto.Types.DataType.String => Const.FromSpan(),
                    //TensorProto.Types.DataType.Uint32 => Const.FromSpan(),
                    //TensorProto.Types.DataType.Uint64 => Const.FromSpan<ulong>(tensor.Uint64Data.ToArray(), shape),
                    //TensorProto.Types.DataType.Uint8 => Const.FromSpan(),
                    _ => throw new NotSupportedException($"Not supported onnx constant data type{dt}")
                };
            }
        }
        public Module Import()
        {
            _constTensors = _graph.Initializer
                .ToDictionary(tensor => tensor.Name, tensor => tensor);

            _outputTensors = _graph.Input
                .Filter(n => !_constTensors.ContainsKey(n.Name))
                .ToDictionary(n => n.Name, n => (Expr) new Var(n.Name, GetIRType(n)));

            var createdInputs = _outputTensors.Values.ToArray();
            _graph.Node.ToList().ForEach(Visit);

            var outputs = _graph.Output.Select(o => _outputTensors[o.Name]).ToArray();

            return MakeMainModule(outputs, createdInputs);
        }

        private Module MakeMainModule(Expr[] body, IRArray<Expr> parameter)
        {
            var outputTuple = new IR.Tuple(ImmutableArray.Create(body));
            var mainFunc = new Function("main", outputTuple, parameter);
            var module = new Module();
            module.Add(mainFunc);
            module.Entry = mainFunc;
            return module;
        }

        public Shape GetShape(ValueInfoProto v)
        {
            var shape = v.Type.TensorType.Shape.Dim.Select(x => x.DimValue);
            return new Shape(shape);
        }

        public Shape GetShape(TensorProto tensor)
        {
            return new Shape(tensor.Dims);
        }

        public TensorType GetIRType(ValueInfoProto v)
        {
            return new TensorType(GetDataType(v), GetShape(v));
        }        
        
        public TensorType GetIRType(TensorProto v)
        {
            return new TensorType(GetDataType(v), GetShape(v));
        }
        
        private Expr GetInputExpr(NodeProto n, int index)
        {
            // todo:is null?
            var id = n.Input[index];
            if (_outputTensors.TryGetValue(id, out var expr))
            {
                return expr;
            }
            return _graph.Initializer
                    .Find(x => x.Name == id)
                    .Match(
                        GetConst,
                        () => throw new InvalidDataException($"Cannot load tensor data (tensor:{id})."));
        }

        private Expr GetSingleInputExpr(NodeProto n)
        {
            return GetInputExpr(n, 0);
        }
        
        private (Expr, Expr) GetInputExprs(NodeProto n, int index0, int index1)
        {
            return (GetInputExpr(n, index0), GetInputExpr(n, index1));
        }

        private Option<Expr> GetOptionInputExpr(NodeProto n, int index)
        {
            if (n.Input.Count <= index)
            {
                return Option.None;
            }
            return Option.Some(GetInputExpr(n, index));
        }

        private (Option<Expr>, Option<Expr>) GetOptionInputExprs(NodeProto n, int index0, int index1)
        {
            return (GetOptionInputExpr(n, index0), GetOptionInputExpr(n, 1));
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
        
        private void Visit(NodeProto op)
        {
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
                // "DequantizeLinear" => VisitDequantizeLinear(op),
                "Div" => VisitBinary(op, BinaryOp.Div),
                "Dropout" => VisitDropout(op),
                "Elu" => VisitElu(op),
                "Exp" => VisitUnary(op, UnaryOp.Exp),
                "Expand" => VisitExpand(op),
                "Flatten" => VisitFlatten(op),
                "Floor" => VisitUnary(op, UnaryOp.Floor),
                "Gather" => VisitGather(op),
                "GatherND" => VisitGatherND(op),
                "Gemm" => VisitGemm(op),
                "GlobalAveragePool" => VisitReduceWindow2D(op, ReduceOp.Mean, float.MinValue, true),
                "GlobalMaxPool" => VisitReduceWindow2D(op, ReduceOp.Max, float.MinValue, true),
                "Hardmax" => VisitHardmax(op),
                "HardSigmoid" => VisitHardSigmoid(op),
                "HardSwish" => VisitHardSwish(op),
                "Identity" => VisitIdentity(op),
                "InstanceNormalization" => VisitInstanceNormalization(op),
                "LpNormalization" => VisitLpNormalization(op),
                "LeakyRelu" => VisitLeakyRelu(op),
                "Log" => VisitUnary(op, UnaryOp.Log),
                "LogSoftmax" => VisitLogSoftmax(op),
                "LRN" => VisitLRN(op),
                "MatMul" => VisitMatMul(op),
                "MaxPool" => VisitReduceWindow2D(op, ReduceOp.Max, float.MinValue),
                "Max" => VisitBinary(op, BinaryOp.Max),
                "Min" => VisitBinary(op, BinaryOp.Min),
                "Mul" => VisitBinary(op, BinaryOp.Mul),
                "Neg" => VisitUnary(op, UnaryOp.Neg),
                // "OneHot" => VisitOneHot(op),
                "Pad" => VisitPad(op),
                "Pow" => VisitBinary(op, BinaryOp.Pow),
                "PRelu" => VisitPRelu(op),
                // "QuantizeLinear" => VisitQuantizeLinear(op),
                "RandomNormal" => VisitRandomNormal(op),
                "RandomNormalLike" => VisitRandomNormalLike(op),
                "RandomUniform" => VisitRandomUniform(op),
                "RandomUniformLike" => VisitRandomUniformLike(op),
                "ReduceL1" => VisitReduceL1(op),
                "ReduceL2" => VisitReduceL2(op),
                "ReduceLogSum" => VisitReduceLogSum(op),
                "ReduceLogSumExp" => VisitReduceLogSumExp(op),
                "ReduceMax" => VisitReduce(op, ReduceOp.Max, float.MinValue),
                "ReduceMean" => VisitReduce(op, ReduceOp.Mean, 0f),
                "ReduceMin" => VisitReduce(op, ReduceOp.Min, float.MaxValue),
                "ReduceSum" => VisitReduce(op, ReduceOp.Sum, 0f),
                "ReduceSumSquare" => VisitReduceSumSquare(op),
                "Relu" => VisitRelu(op),
                "Reshape" => VisitReshape(op),
                // "Resize" => VisitResize(op),
                "Round" => VisitUnary(op, UnaryOp.Round),
                "Selu" => VisitSelu(op),
                // "Shape" => VisitShape(op),
                "Sin" => VisitUnary(op, UnaryOp.Sin),
                "Sinh" => VisitUnary(op, UnaryOp.Sinh),
                "Sigmoid" => VisitSigmoid(op),
                // "Size" => VisitSize(op),
                // "Slice" => VisitSlice(op),
                // "Softmax" => VisitSoftmax(op),
                // "Softplus" => VisitSoftplus(op),
                // "Softsign" => VisitSoftsign(op),
                // "SpaceToDepth" => VisitSpaceToDepth(op),
                // "Split" => VisitSplit(op),
                "Sqrt" => VisitUnary(op, UnaryOp.Sqrt),
                // "Squeeze" => VisitSqueeze(op),
                "Sub" => VisitBinary(op, BinaryOp.Sub),
                // "Sum" => VisitSum(op),
                "Tanh" => VisitUnary(op, UnaryOp.Tanh),
                // "Transpose" => VisitTranspose(op),
                // "Upsample" => VisitUpsample(op),
                // "Unsqueeze" => VisitUnsqueeze(op),
                _ => throw new NotSupportedException($"Not Supported onnx op {op.OpType}")
            };

            if (output is Expr expr)
            {
                Debug.Assert(op.Output.Count == 1, "Op outputs length should be 1.");
                _outputTensors.Add(op.Output[0], expr);
            }
            else if (output is IReadOnlyList<Expr> exprList)
            {
                Debug.Assert(op.Output.Count == exprList.Count, $"Op outputs length should be {op.Output.Count}.");
                for (int i = 0; i < op.Output.Count; i++)
                {
                    _outputTensors.Add(op.Output[i], exprList[i]);
                }
            }
            else
            {
                throw new InvalidOperationException("Visit result is not expression(s).");
            }
        }
    }
}
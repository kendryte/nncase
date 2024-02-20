// Copyright (c) Canaan Inc. All rights reserved.
// Licensed under the Apache license. See LICENSE file in the project root for full license information.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Reactive;
using System.Text;
using System.Threading.Tasks;
using Microsoft.Toolkit.HighPerformance;
using NetFabric.Hyperlinq;
using Nncase.Diagnostics;
using Nncase.IR;
using Nncase.IR.Ncnn;
using Nncase.IR.Tensors;
using Nncase.Utilities;
using static Nncase.CodeGen.CodeGenDumper;

namespace Nncase.CodeGen.Ncnn;

/// <summary>
/// Ncnn function builder.
/// </summary>
internal class NcnnFunctionBuilder : FunctionBuilder
{
    private readonly NcnnEmitter _emitter;
    private string[]? _inputs;
    private string[]? _outputs;

    public NcnnFunctionBuilder(uint id, SectionManager sectionManager)
        : base(id, sectionManager)
    {
        _emitter = new NcnnEmitter(sectionManager.GetWriter(WellknownSectionNames.Rdata));
    }

    protected override ILinkableFunction CreateLinkableFunction(uint id, BaseFunction callable, IReadOnlyList<FunctionRef> functionRefs, Stream text)
    {
        return new NcnnLinkableFunction(id, callable, functionRefs, text, _inputs!, _outputs!);
    }

    protected override void Compile(BaseFunction callable)
    {
        var visitor = new CodeGenVisitor(_emitter);
        _outputs = visitor.Visit(callable).Split(',');
        _inputs = visitor.Inputs.ToArray();
    }

    protected override void WriteText()
    {
        _emitter.SaveParam(TextWriter.BaseStream);

        using (var fileStream = File.Create(Directory.GetCurrentDirectory() + "/ncnn.param"))
        {
            TextWriter.BaseStream.Seek(0, SeekOrigin.Begin);
            TextWriter.BaseStream.CopyTo(fileStream);
        }

        _emitter.SaveBin();
    }

    private class CodeGenVisitor : ExprVisitor<string, Unit>
    {
        private readonly NcnnEmitter _emitter;

        private int _layerId;

        public CodeGenVisitor(NcnnEmitter emitter)
        {
            _emitter = emitter;
        }

        public List<string> Inputs { get; } = new();

        public List<string> Outputs { get; } = new();

        protected override string VisitLeafVar(Var expr)
        {
            var name = GetNextName();
            _emitter.Input(name);
            Inputs.Add(name);
            return name;
        }

        protected override string VisitLeafFunction(Function expr) => ExprMemo[expr.Body];

        protected override string VisitLeafOp(Op expr) => string.Empty;

        protected override string VisitLeafTuple(IR.Tuple expr) => StringUtility.Join(",", expr.Fields.AsValueEnumerable().Select(x => ExprMemo[x]));

        protected override string VisitLeafCall(Call expr)
        {
            var names = new List<string> { GetNextName() };
            switch (expr.Target)
            {
                case NcnnSoftmax op:
                    _emitter.Softmax(names[0], ExprMemo[expr.Arguments[0]], op.Axis);
                    break;
                case NcnnUnary op:
                    _emitter.Unary(names[0], ExprMemo[expr.Arguments[0]], op.OpType);
                    break;
                case NcnnBatchNorm op:
                    _emitter.BatchNorm(names[0], ExprMemo[expr.Arguments[0]], op.Channels, op.Eps, op.SlopeData, op.MeanData, op.VarData, op.BiasData);
                    break;
                case NcnnBinary op:
                    string[]? inString = op.LorR switch
                    {
                        0 => new string[] { ExprMemo[expr.Arguments[0]], ExprMemo[expr.Arguments[1]] },
                        1 => new string[] { string.Empty, ExprMemo[expr.Arguments[0]] },
                        2 => new string[] { ExprMemo[expr.Arguments[0]], string.Empty },
                        _ => throw new NotImplementedException("Not found binary emmiter."),
                    };
                    _emitter.Binary(names[0], inString[0], inString[1], op.OpType, op.LorR, op.ConstInput, op.ConstShape);
                    break;
                case NcnnCelu op:
                    _emitter.Celu(names[0], ExprMemo[expr.Arguments[0]], op.Alpha);
                    break;
                case NcnnClip op:
                    _emitter.Clip(names[0], ExprMemo[expr.Arguments[0]], op.Min, op.Max);
                    break;
                case NcnnConcat op:
                    List<string> in_ = new();
                    var t = (IR.Tuple)expr.Arguments[0];
                    for (int i = 0; i < t.Fields.Length; i++)
                    {
                        in_.Add(ExprMemo[t.Fields[i]]);
                    }

                    _emitter.Concat(names[0], in_.ToArray(), op.Axis);
                    break;
                case NcnnConv op:
                    _emitter.Conv(names[0], ExprMemo[expr.Arguments[0]], op.WeightData, op.BiasData, op.NumOutput, op.KernelW, op.KernelH, op.DilationW, op.DilationH, op.StrideW, op.StrideH, op.PadLeft, op.PadTop, op.PadRight, op.PadBottom, op.BiasTerm, op.WeightDataSize, op.Int8ScaleTerm, op.ActivationType, op.ActivationParams, op.PadValue, op.DynamicWeight);
                    break;
                case NcnnCumsum op:
                    _emitter.Cumsum(names[0], ExprMemo[expr.Arguments[0]], op.Axis);
                    break;
                case NcnnElu op:
                    _emitter.Elu(names[0], ExprMemo[expr.Arguments[0]], op.Alpha);
                    break;
                case NcnnErf:
                    _emitter.Erf(names[0], ExprMemo[expr.Arguments[0]]);
                    break;
                case NcnnHardSigmoid op:
                    _emitter.HardSigmoid(names[0], ExprMemo[expr.Arguments[0]], op.Alpha, op.Beta);
                    break;
                case NcnnHardSwish op:
                    _emitter.HardSwish(names[0], ExprMemo[expr.Arguments[0]], op.Alpha, op.Beta);
                    break;
                case NcnnInstanceNorm op:
                    _emitter.InstanceNorm(names[0], ExprMemo[expr.Arguments[0]], op.Channels, op.Eps, op.Affine, op.GammaData, op.BetaData);
                    break;
                case NcnnLRN op:
                    _emitter.LRN(names[0], ExprMemo[expr.Arguments[0]], op.Alpha, op.Beta, op.Bias, op.Size);
                    break;
                case NcnnLSTM op:
                    for (int i = 1; i < op.OutputSize; i++)
                    {
                        var a = GetNextName();
                        Console.WriteLine($"{i} = {a}");
                        names.Add(a);
                    }

                    _emitter.LSTM(names.ToArray(), ExprMemo[expr.Arguments[0]], op.HiddenSize, op.WeightDataSize, op.Direction, op.W, op.R, op.B);
                    break;
                case NcnnPadding op:
                    _emitter.Padding(names.ToArray(), ExprMemo[expr.Arguments[0]], op.Top, op.Bottom, op.Left, op.Right, op.Type, op.Value, op.Front, op.Behind);
                    break;
                case NcnnPooling op:
                    _emitter.Pooling(names.ToArray(), ExprMemo[expr.Arguments[0]], op.Args);
                    break;
                case NcnnPReLU op:
                    _emitter.PReLU(names.ToArray(), ExprMemo[expr.Arguments[0]], op.Slope);
                    break;
                case NcnnReduction op:
                    _emitter.Reduction(names.ToArray(), ExprMemo[expr.Arguments[0]], op.Args);
                    break;
                case NcnnReshape op:
                    _emitter.Reshape(names.ToArray(), ExprMemo[expr.Arguments[0]], op.Shape);
                    break;
                case NcnnSELU op:
                    _emitter.SELU(names.ToArray(), ExprMemo[expr.Arguments[0]], op.Alpha, op.Gamma);
                    break;
                case NcnnSigmoid:
                    _emitter.Sigmoid(names.ToArray(), ExprMemo[expr.Arguments[0]]);
                    break;
                case NcnnCrop op:
                    _emitter.Crop(names.ToArray(), ExprMemo[expr.Arguments[0]], op.Args);
                    break;
                case NcnnSoftplus:
                    _emitter.Softplus(names.ToArray(), ExprMemo[expr.Arguments[0]]);
                    break;
                default:
                    throw new NotSupportedException();
            }

            // serialize outputs to string.
            string output = string.Join(",", names.Select(x => x.ToString()));
            return output;
        }

        private string GetNextName() => $"layer{_layerId++}";
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using NnCase.Evaluation;
using NnCase.IR;
using NnCase.IR.Operators;
using NnCase.Targets.CPU.IR.Operators;
using NnCase.Transforms;

namespace NnCase.Targets.CPU.Transforms
{
    public class CPUQuantizedConv2DTransform : Transform
    {
        private readonly Quantizer _quantizer;

        protected override bool SkipSelfContainedCheck => true;

        public CPUQuantizedConv2DTransform(Quantizer quantizer)
        {
            _quantizer = quantizer;
        }

        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is FakeQuantize q
                && NodeTreeHelper.TryGetDirectChild<CPUConv2D>(q, out var conv2d)
                && NodeTreeHelper.TryGetDirectChild<FakeDequantize>(conv2d, out var deq))
            {
                context.Inputs.Add(q.Input);
                context.Outputs.Add(deq.Output);
                context.MatchedNodes.Add(q);
                context.MatchedNodes.Add(conv2d);
                context.MatchedNodes.Add(deq);
                return true;
            }

            return false;
        }

        public override void Process(TransformContext context)
        {
            var output = context.Inputs[0].Connection;
            var inputs = context.Outputs[0].Connections;
            var fq = (FakeQuantize)context.MatchedNodes[0];
            var oldConv2D = (CPUConv2D)context.MatchedNodes[1];
            var fdeq = (FakeDequantize)context.MatchedNodes[2];

            var iqParam = _quantizer.GetQuantizationParam(_quantizer.Get(fq.Output), 8);
            (var wqParam, var qWeights) = QuantizeWeights(oldConv2D);
            var yqParam = _quantizer.GetQuantizationParam(_quantizer.Get(fdeq.Output), 8);
            var sa = iqParam.Scale * wqParam.Scale;
            var so = yqParam.Scale / sa;
            var qBias = QuantizeBias(oldConv2D, sa);
            var oMul = _quantizer.GetFixedMul(so, 32, 31, true);

            var q = context.Graph.AddNode(new Quantize(fq.Input.Shape, iqParam));
            var conv2d = context.Graph.AddNode(new CPUQuantizedConv2D(q.Output.Shape, qWeights, qBias, oldConv2D.PaddingH, oldConv2D.PaddingW, oldConv2D.StrideH, oldConv2D.StrideW, oldConv2D.DilationH, oldConv2D.DilationW, iqParam.ZeroPoint, wqParam.ZeroPoint, oMul.RoundedMul, oMul.Shift, yqParam.ZeroPoint));
            var deq = context.Graph.AddNode(new Dequantize(conv2d.Output.Shape, yqParam));
            q.Input.Connect(output);
            conv2d.Input.Connect(q.Output);
            deq.Input.Connect(conv2d.Output);

            foreach (var input in inputs.ToList())
                input.Connect(deq.Output);
        }

        private DenseTensor<int> QuantizeBias(CPUConv2D conv2d, float sa)
        {
            var newBias = new DenseTensor<int>(conv2d.Bias.Dimensions);
            var src = conv2d.Bias.Buffer.Span;
            var dest = newBias.Buffer.Span;

            for (int i = 0; i < dest.Length; i++)
            {
                dest[i] = (int)Math.Round(src[i] * sa);
            }

            return newBias;
        }

        private (QuantizationParam wqParam, DenseTensor<byte> qWeights) QuantizeWeights(CPUConv2D conv2d)
        {
            var newWeights = new DenseTensor<byte>(conv2d.Weights.Dimensions);
            var src = conv2d.Weights.Buffer.Span;
            var dest = newWeights.Buffer.Span;
            var quantParam = _quantizer.GetQuantizationParam(_quantizer.GetRange(src), 8);

            for (int i = 0; i < dest.Length; i++)
            {
                dest[i] = (byte)Math.Clamp((int)Math.Round(src[i] * quantParam.Scale + quantParam.ZeroPoint), 0, 255);
            }

            return (quantParam, newWeights);
        }
    }
}

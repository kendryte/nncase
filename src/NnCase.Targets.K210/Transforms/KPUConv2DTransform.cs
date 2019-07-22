using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Numerics.Tensors;
using System.Text;
using NnCase.Evaluation;
using NnCase.IR;
using NnCase.IR.Operators;
using NnCase.Targets.K210.IR;
using NnCase.Targets.K210.IR.FakeOperators;
using NnCase.Transforms;

namespace NnCase.Targets.K210.Transforms
{
    public class KPUConv2DTransform : Transform
    {
        private readonly Quantizer _quantizer;

        protected override bool SkipSelfContainedCheck => true;

        public KPUConv2DTransform(Quantizer quantizer)
        {
            _quantizer = quantizer;
        }

        protected override bool OnTryMatch(Node node, TransformContext context)
        {
            if (node is FakeQuantize q
                && NodeTreeHelper.TryGetDirectChild<KPUFakeConv2D>(q, out var conv2d)
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
            var oldConv2D = (KPUFakeConv2D)context.MatchedNodes[1];
            var fdeq = (FakeDequantize)context.MatchedNodes[2];

            var iqParam = _quantizer.GetQuantizationParam(_quantizer.Get(fq.Output), 8);
            (var wqParam, var qWeights) = QuantizeWeights(oldConv2D);
            var yqParam = _quantizer.GetQuantizationParam(_quantizer.Get(fdeq.Output), 8);
            var sa = iqParam.Scale * wqParam.Scale;
            (var batchNorm, var activation) = QuantizeBiasAndOutput(oldConv2D, sa, yqParam);

            var filter = KPUShapeUtility.GetKPUFilterSize(oldConv2D.FilterType);

            var q = context.Graph.AddNode(new Quantize(fq.Input.Shape, iqParam));
            var upload = context.Graph.AddNode(new KPUUpload(q.Input.Shape));
            var conv2d = context.Graph.AddNode(new KPUConv2D(upload.Output.Shape, oldConv2D.IsDepthwise, oldConv2D.FilterType, oldConv2D.PoolType, qWeights, (byte)iqParam.ZeroPoint, -wqParam.ZeroPoint, 0, -iqParam.ZeroPoint, 0, filter * filter * wqParam.ZeroPoint * iqParam.ZeroPoint, batchNorm, activation, false));
            var download = context.Graph.AddNode(new KPUDownload(conv2d.KPUOutput.Shape));
            var deq = context.Graph.AddNode(new Dequantize(download.Output.Shape, yqParam));
            q.Input.Connect(output);
            upload.Input.Connect(q.Output);
            conv2d.Input.Connect(upload.Output);
            download.Input.Connect(conv2d.KPUOutput);
            deq.Input.Connect(download.Output);

            foreach (var input in inputs.ToList())
                input.Connect(deq.Output);
        }

        private (KPUBatchNormSegment[] batchNorm, KPUActivationSegment[] activation) QuantizeBiasAndOutput(KPUFakeConv2D conv2d, float sa, in QuantizationParam yqParam)
        {
            var batchNorm = new KPUBatchNormSegment[conv2d.Output.Shape[1]];
            var bias = conv2d.Bias.Buffer.Span;

            var so = yqParam.Scale / sa;
            var bnMul = _quantizer.GetFixedMul(so, 22, 255, true);
            var upscale = bnMul.Shift - 15;
            Debug.Assert(upscale >= 0);
            var postMul = bnMul.RoundedMul / bnMul.Mul * MathF.Pow(2, upscale);

            for (int i = 0; i < bias.Length; i++)
            {
                var b = bias[i];
                batchNorm[i] = new KPUBatchNormSegment
                {
                    Mul = bnMul.RoundedMul,
                    Shift = 15,
                    Add = (int)Math.Round((b + yqParam.Scale * yqParam.ZeroPoint) / yqParam.Scale * postMul)
                };
            }

            return (batchNorm, QuantizeActivation(conv2d, yqParam, postMul));
        }

        private KPUActivationSegment[] QuantizeActivation(KPUFakeConv2D conv2d, in QuantizationParam yqParam, float postMul)
        {
            var activation = new KPUActivationSegment[16];

            var starts = new long[]
            {
                0x800000000, 0xf7d4cf4b8, 0xf8ed5a20c, 0xfa05e4f60,
                0xfb2e05baa, 0xfc46908fe, 0xfd5f1b652, 0xfe77a63a6,
                0xff9fc6ff0, 0xfffd4a9b7, 0, 0x7FFFFFFF0,
                0x7FFFFFFF1, 0x7FFFFFFF2, 0x7FFFFFFF3, 0x7FFFFFFF4
            };

            for (int i = 0; i < starts.Length; i++)
            {
                ref var param = ref activation[i];
                param.StartX = starts[i];

                if (i == 10)
                {
                    var mul = _quantizer.GetFixedMul(1 / postMul, 16, 20, true);
                    param.Mul = mul.RoundedMul;
                    param.Shift = mul.Shift;
                }
            }

            return activation;
        }

        private (QuantizationParam wqParam, DenseTensor<byte> qWeights) QuantizeWeights(KPUFakeConv2D conv2d)
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

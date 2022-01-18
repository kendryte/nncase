using System;
using System.Linq;
using System.Collections.Generic;
using NetFabric.Hyperlinq;
using Nncase.IR.NN;
using TorchSharp;
using Nncase.IR;

using torchF = TorchSharp.torch.nn.functional;

namespace Nncase.Evaluator.Ops
{
    public class Conv2DEvaluator : IEvaluator<Conv2D>
    {
        public Const Visit(EvaluatorContext context, Conv2D conv)
        {
            var input = context.GetTorchArgument(conv, Conv2D.Input);
            var weights = context.GetTorchArgument(conv, Conv2D.Weights);
            var bias = context.GetTorchArgument(conv, Conv2D.Bias);
            var stride = context.GetArgumentConst(conv, Conv2D.Stride).ToTensor<long>();
            // [w:[left right] h:[top bottom]]
            var pad = context.GetArgumentConst(conv, Conv2D.Padding).ToTensor<long>();
            var dilation = context.GetArgumentConst(conv, Conv2D.Dilation).ToTensor<long>();
            var groups = context.GetArgumentConst(conv, Conv2D.Groups).ToScalar<long>();
            if (conv.PadMode != PadMode.Constant)
            {
                throw new NotImplementedException($"Conv2D with {conv.PadMode}!");
            }
            // pad in TorchSharp will reorder
            // when pad.Count == 4, [0, 2, 1, 3]
            // order should be passed: left top right bottom
            var afterPad = torchF.pad(input, new long[] { pad[0, 0], pad[1, 0], pad[0, 1], pad[1, 1] });
            return torchF.conv2d(
                afterPad,
              weights, bias,
              strides: new long[] { stride[0], stride[1] },
              dilation: new long[] { dilation[0], dilation[1] },
              groups: groups).ToConst();
        }
    }
}